#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>

#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"
#include "ThreadPool.h"

// Tile-based multithreading and vertex cache (always on)
static const int MT_POOL_SIZE = 11;
static const int MT_TILE_W = 256;
static const int MT_TILE_H = 256;

// Forward declaration for the base (no vertex cache, no MT) branch of switchRender.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L);

// Transform all vertices for current frame per mesh into vertexCaches[i].
// Each vertex is transformed once per frame; lookup by vertex index.
static void fillFrameVertexCaches(Renderer& renderer, std::vector<Mesh*>& scene, matrix& camera,
    std::vector<std::vector<Vertex>>& vertexCaches) {
    const float w = static_cast<float>(renderer.canvas.getWidth());
    const float h = static_cast<float>(renderer.canvas.getHeight());
    vertexCaches.resize(scene.size());
    for (size_t mi = 0; mi < scene.size(); mi++) {
        Mesh* mesh = scene[mi];
        matrix p = renderer.perspective * camera * mesh->world;
        std::vector<Vertex>& cache = vertexCaches[mi];
        cache.resize(mesh->vertices.size());
        for (size_t vi = 0; vi < mesh->vertices.size(); vi++) {
            Vertex& out = cache[vi];
            out.p = p * mesh->vertices[vi].p;
            out.p.divideW();
            out.normal = mesh->world * mesh->vertices[vi].normal;
            out.normal.normalise();
            out.p[0] = (out.p[0] + 1.f) * 0.5f * w;
            out.p[1] = (out.p[1] + 1.f) * 0.5f * h;
            out.p[1] = h - out.p[1];
            out.rgb = mesh->vertices[vi].rgb;
        }
    }
}

// Build this frame's triangle list from vertex cache, read-only by triangle index; no duplicate vertex transform.
static void buildAllTrianglesFromCache(std::vector<Mesh*>& scene,
    std::vector<std::vector<Vertex>>& vertexCaches, std::vector<triangle>& allTriangles) {
    size_t totalTris = 0;
    for (size_t mi = 0; mi < scene.size(); mi++)
        totalTris += scene[mi]->triangles.size();
    allTriangles.clear();
    allTriangles.reserve(totalTris);
    for (size_t mi = 0; mi < scene.size(); mi++) {
        Mesh* mesh = scene[mi];
        std::vector<Vertex>& cache = vertexCaches[mi];
        for (triIndices& ind : mesh->triangles) {
            const Vertex& v0 = cache[ind.v[0]];
            const Vertex& v1 = cache[ind.v[1]];
            const Vertex& v2 = cache[ind.v[2]];
            if (fabs(v0.p[2]) > 1.0f || fabs(v1.p[2]) > 1.0f || fabs(v2.p[2]) > 1.0f) continue;
            allTriangles.emplace_back(v0, v1, v2, mesh);
        }
    }
}

// Triangle–tile binning: record per tile the triangle indices overlapping that tile.
static void buildTrianglesPerTile(Renderer& renderer, std::vector<triangle>& allTriangles,
    int numTilesX, int numTilesY, int w, int h,
    std::vector<std::vector<size_t>>& trianglesPerTile) {
    const int numTiles = numTilesX * numTilesY;
    const size_t nTri = allTriangles.size();
    if (trianglesPerTile.size() != static_cast<size_t>(numTiles))
        trianglesPerTile.resize(static_cast<size_t>(numTiles));
    const size_t reservePerTile = (nTri / (size_t)numTiles) + 32u;
    for (int t = 0; t < numTiles; t++) {
        trianglesPerTile[t].clear();
        trianglesPerTile[t].reserve(reservePerTile);
    }
    for (size_t i = 0; i < nTri; i++) {
        vec2D minV, maxV;
        allTriangles[i].getBoundsWindow(renderer.canvas, minV, maxV);
        const int tileMinX = (std::max)(0, (int)(minV.x) / MT_TILE_W);
        const int tileMaxX = (std::min)(numTilesX - 1, (int)(maxV.x) / MT_TILE_W);
        const int tileMinY = (std::max)(0, (int)(minV.y) / MT_TILE_H);
        const int tileMaxY = (std::min)(numTilesY - 1, (int)(maxV.y) / MT_TILE_H);
        for (int ty = tileMinY; ty <= tileMaxY; ty++) {
            const int row = ty * numTilesX;
            for (int tx = tileMinX; tx <= tileMaxX; tx++)
                trianglesPerTile[row + tx].push_back(i);
        }
    }
}

// Tile job context: single set of refs, job carries only tileIndex to reduce capture and allocation.
struct TileJobContext {
    Renderer* r;
    Light* L;
    std::vector<triangle>* tris;
    std::vector<std::vector<size_t>>* perTile;
    int numTilesX, w, h;
};

static void runTileJobFromContext(const TileJobContext* ctx, int tileIndex) {
    const int tx = tileIndex % ctx->numTilesX;
    const int ty = tileIndex / ctx->numTilesX;
    const int startX = tx * MT_TILE_W;
    const int startY = ty * MT_TILE_H;
    const int endX = (std::min)(startX + MT_TILE_W, ctx->w);
    const int endY = (std::min)(startY + MT_TILE_H, ctx->h);
    const std::vector<size_t>& list = (*ctx->perTile)[tileIndex];
    std::vector<triangle>& allTriangles = *ctx->tris;
    for (size_t i : list)
        allTriangles[i].drawParallel(*ctx->r, *ctx->L, startX, startY, endX, endY);
}

// Thread pool singleton, created on first use then resident.
static ThreadPool& getTilePool() {
    static ThreadPool pool(MT_POOL_SIZE);
    return pool;
}

static void switchRender(Renderer& renderer, std::vector<Mesh*>& scene, matrix& camera, Light& L) {
    std::vector<std::vector<Vertex>> vertexCaches;
    fillFrameVertexCaches(renderer, scene, camera, vertexCaches);

    std::vector<triangle> allTriangles;
    buildAllTrianglesFromCache(scene, vertexCaches, allTriangles);

    const int w = static_cast<int>(renderer.canvas.getWidth());
    const int h = static_cast<int>(renderer.canvas.getHeight());
    const int numTilesX = (w + MT_TILE_W - 1) / MT_TILE_W;
    const int numTilesY = (h + MT_TILE_H - 1) / MT_TILE_H;
    const int numTiles = numTilesX * numTilesY;

    static std::vector<std::vector<size_t>> trianglesPerTile;
    buildTrianglesPerTile(renderer, allTriangles, numTilesX, numTilesY, w, h, trianglesPerTile);

    TileJobContext ctx = { &renderer, &L, &allTriangles, &trianglesPerTile, numTilesX, w, h };
    static std::vector<std::function<void()>> jobs;
    jobs.clear();
    jobs.reserve(static_cast<size_t>(numTiles));
    for (int t = 0; t < numTiles; t++) {
        const int tileIndex = t;
        jobs.push_back([&ctx, tileIndex]() { runTileJobFromContext(&ctx, tileIndex); });
    }

    ThreadPool& pool = getTilePool();
    pool.submitBatch(jobs);
    pool.waitIdle();
}

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal; 
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}

// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // camera is just a matrix
    matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
   // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        // Render each object in the scene
        switchRender(renderer, scene, camera, L);

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };

    bool running = true;

    std::vector<Mesh*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        switchRender(renderer, scene, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        switchRender(renderer, scene, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Scene 3: six spheres orbiting in XY plane; output format same as Scene 1/2 (number : ms every 2 rotations).
void scene3() {
    Renderer renderer;
    matrix camera = matrix::makeTranslation(0.f, 0.f, -5.5f);
    Light L{ vec4(0.3f, 1.f, 0.5f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };

    std::vector<Mesh*> scene;
    const int numOrbit = 6;
    const float orbitRadius = 2.2f;
    const float orbitZ = -0.6f;
    const float orbitSpeed = 0.03f;
    const float oneTurnRad = 2.0f * static_cast<float>(M_PI);
    const int lat = 14, lon = 28;

    // --- 6 spheres orbiting in XY plane (parallel to screen) ---
    for (int i = 0; i < numOrbit; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeSphere(0.45f, lat, lon);
        m->setColour(colour(0.6f + 0.05f * i, 0.75f, 0.9f - 0.05f * i), 0.6f, 0.75f);
        scene.push_back(m);
    }

    // 10x10 static grid behind orbit, center spacing = diameter + a bit
    const float gridZ = -1.5f;
    const float sphereRadius = 0.45f;
    const float gridStep = 2.f * sphereRadius + 0.02f;  // Center spacing = diameter + a bit, just touching.
    const float gridHalf = (10 - 1) * gridStep * 0.5f;
    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 10; col++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeSphere(sphereRadius, lat, lon);
            m->setColour(colour(0.5f + 0.02f * row, 0.7f, 0.75f - 0.02f * col), 0.55f, 0.7f);
            float x = -gridHalf + gridStep * col;
            float y = gridHalf - gridStep * row;
            m->world = matrix::makeTranslation(x, y, gridZ);
            scene.push_back(m);
        }
    }

    const int idxOrbitStart = 0;
    float orbitAngle = 0.f;
    float totalRotated = 0.f;
    int lastReportedTurns = 0;
    int lastPrintedAtTurn = 0;  // print when we complete 2 more rotations (same logic as Scene 1/2)
    auto segmentStart = std::chrono::high_resolution_clock::now();
    bool running = true;

    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0.f, 0.f, -5.5f);

        for (int i = 0; i < numOrbit; i++) {
            float angle = orbitAngle + static_cast<float>(i) * (2.0f * static_cast<float>(M_PI) / numOrbit);
            float x = orbitRadius * std::cos(angle);
            float y = orbitRadius * std::sin(angle);
            scene[idxOrbitStart + i]->world = matrix::makeTranslation(x, y, orbitZ);
        }
        orbitAngle -= orbitSpeed;
        totalRotated += orbitSpeed;

        int completedTurns = static_cast<int>(totalRotated / oneTurnRad);
        if (completedTurns > lastReportedTurns) {
            auto now = std::chrono::high_resolution_clock::now();
            lastReportedTurns = completedTurns;
            if (completedTurns >= lastPrintedAtTurn + 2) {
                lastPrintedAtTurn += 2;
                std::cout << (lastPrintedAtTurn / 2) << " :" << std::chrono::duration<double, std::milli>(now - segmentStart).count() << "ms\n";
                segmentStart = now;
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        switchRender(renderer, scene, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Entry point of the application.
// No input variables
int main() {
    // Uncomment the desired scene function to run
   scene1();
    //scene2();
    //scene3();
    //sceneTest();

    return 0;
}