#pragma once
#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

// Simple support class for a 2D vector
class vec2D {
public:
    float x, y;

    // Default constructor initializes both components to 0
    vec2D() { x = y = 0.f; };

    // Constructor initializes components with given values
    vec2D(float _x, float _y) : x(_x), y(_y) {}

    // Constructor initializes components from a vec4
    vec2D(vec4 v) {
        x = v[0];
        y = v[1];
    }

    // Display the vector components
    void display() { std::cout << x << '\t' << y << std::endl; }

    // Overloaded subtraction operator for vector subtraction
    vec2D operator- (vec2D& v) {
        vec2D q;
        q.x = x - v.x;
        q.y = y - v.y;
        return q;
    }
};

// Class representing a triangle for rendering purposes
class triangle {
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    Mesh* parentMesh;  // for drawParallel tile path, nullptr when not used

public:
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        parentMesh = nullptr;
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);
        area = std::fabs(e1.x * e2.y - e1.y * e2.x);
    }

    // Constructor with mesh pointer for drawParallel; uses parentMesh->kd
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3, Mesh* mesh) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        parentMesh = mesh;
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);
        area = std::fabs(e1.x * e2.y - e1.y * e2.x);
    }

    // For single-threaded vertex-cache path: get mesh material when parentMesh is set
    float getKa() const { return parentMesh ? parentMesh->ka : 0.f; }
    float getKd() const { return parentMesh ? parentMesh->kd : 0.f; }

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    float getC(vec2D v1, vec2D v2, vec2D p) {
        vec2D e = v2 - v1;
        vec2D q = p - v1;
        return q.y * e.x - q.x * e.y;
    }

    inline __m256 getC_SIMD(__m256 v1x, __m256 v1y, __m256 v2x, __m256 v2y, __m256 px, __m256 py) {
        __m256 ex = _mm256_sub_ps(v2x, v1x);
        __m256 ey = _mm256_sub_ps(v2y, v1y);
        __m256 qx = _mm256_sub_ps(px, v1x);
        __m256 qy = _mm256_sub_ps(py, v1y);
        return _mm256_sub_ps(_mm256_mul_ps(qy, ex), _mm256_mul_ps(qx, ey));
    }

    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
        alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
        beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
        gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;

        if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
        return true;
    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }

    __m256 interpolate_SIMD(__m256 alpha, __m256 beta, __m256 gamma,
        __m256 a1, __m256 a2, __m256 a3) {
        return _mm256_add_ps(_mm256_mul_ps(a1, alpha),
            _mm256_add_ps(_mm256_mul_ps(a2, beta),
                _mm256_mul_ps(a3, gamma)));
    }

private:
    void drawTileSIMD(Renderer& renderer, Light& L, float kd, int yStart, int yEnd, int xStart, int xEnd, int tileStartX, int tileEndX) {
        __m256 v0x = _mm256_set1_ps(vec2D(v[0].p).x);
        __m256 v0y = _mm256_set1_ps(vec2D(v[0].p).y);
        __m256 v1x = _mm256_set1_ps(vec2D(v[1].p).x);
        __m256 v1y = _mm256_set1_ps(vec2D(v[1].p).y);
        __m256 v2x = _mm256_set1_ps(vec2D(v[2].p).x);
        __m256 v2y = _mm256_set1_ps(vec2D(v[2].p).y);
        __m256 area_inv = _mm256_set1_ps(1.0f / area);
        for (int y = yStart; y < yEnd; y++) {
            for (int x = xStart; x < xEnd; x += 8) {
                __m256 px = _mm256_setr_ps((float)x, (float)(x + 1), (float)(x + 2), (float)(x + 3), (float)(x + 4), (float)(x + 5), (float)(x + 6), (float)(x + 7));
                __m256 py = _mm256_set1_ps((float)y);
                __m256 alpha = getC_SIMD(v0x, v0y, v1x, v1y, px, py);
                __m256 beta = getC_SIMD(v1x, v1y, v2x, v2y, px, py);
                alpha = _mm256_mul_ps(alpha, area_inv);
                beta = _mm256_mul_ps(beta, area_inv);
                __m256 gamma = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(alpha, beta));
                __m256 mask_inside = _mm256_and_ps(
                    _mm256_cmp_ps(alpha, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
                    _mm256_and_ps(
                        _mm256_cmp_ps(beta, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
                        _mm256_cmp_ps(gamma, _mm256_set1_ps(0.0f), _CMP_GE_OQ)));
                int mask_int = _mm256_movemask_ps(mask_inside);
                __m256 px_start = _mm256_set1_ps((float)tileStartX);
                __m256 px_end = _mm256_set1_ps((float)tileEndX);
                int tile_mask = _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(px, px_start, _CMP_GE_OQ), _mm256_cmp_ps(px, px_end, _CMP_LT_OQ)));
                mask_int &= tile_mask;
                if (mask_int == 0) continue;
                __m256 p0z = _mm256_set1_ps(v[0].p[2]);
                __m256 p1z = _mm256_set1_ps(v[1].p[2]);
                __m256 p2z = _mm256_set1_ps(v[2].p[2]);
                __m256 depth = interpolate_SIMD(beta, gamma, alpha, p0z, p1z, p2z);
                __m256 zbuffer = _mm256_loadu_ps(&renderer.zbuffer(x, y));
                __m256 mask_depth = _mm256_cmp_ps(depth, zbuffer, _CMP_LT_OQ);
                __m256 mask_near = _mm256_cmp_ps(depth, _mm256_set1_ps(0.01f), _CMP_GT_OQ);
                mask_depth = _mm256_and_ps(mask_depth, mask_near);
                mask_int &= _mm256_movemask_ps(mask_depth);
                if (mask_int == 0) continue;
                __m256 normal_x = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].normal.x), _mm256_set1_ps(v[1].normal.x), _mm256_set1_ps(v[2].normal.x));
                __m256 normal_y = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].normal.y), _mm256_set1_ps(v[1].normal.y), _mm256_set1_ps(v[2].normal.y));
                __m256 normal_z = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].normal.z), _mm256_set1_ps(v[1].normal.z), _mm256_set1_ps(v[2].normal.z));
                L.omega_i.normalise();
                __m256 light_dir_x = _mm256_set1_ps(L.omega_i.x);
                __m256 light_dir_y = _mm256_set1_ps(L.omega_i.y);
                __m256 light_dir_z = _mm256_set1_ps(L.omega_i.z);
                __m256 dot_product = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(light_dir_x, normal_x), _mm256_add_ps(_mm256_mul_ps(light_dir_y, normal_y), _mm256_mul_ps(light_dir_z, normal_z))), _mm256_set1_ps(0.0f));
                __m256 color_r = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].rgb.r), _mm256_set1_ps(v[1].rgb.r), _mm256_set1_ps(v[2].rgb.r));
                __m256 color_g = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].rgb.g), _mm256_set1_ps(v[1].rgb.g), _mm256_set1_ps(v[2].rgb.g));
                __m256 color_b = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].rgb.b), _mm256_set1_ps(v[1].rgb.b), _mm256_set1_ps(v[2].rgb.b));
                __m256 Lr = _mm256_set1_ps(L.L.r), Lg = _mm256_set1_ps(L.L.g), Lb = _mm256_set1_ps(L.L.b);
                __m256 Ar = _mm256_set1_ps(L.ambient.r), Ag = _mm256_set1_ps(L.ambient.g), Ab = _mm256_set1_ps(L.ambient.b);
                __m256 kd_ps = _mm256_set1_ps(kd);
                __m256 scale_r = _mm256_add_ps(_mm256_mul_ps(Lr, dot_product), _mm256_mul_ps(Ar, kd_ps));
                __m256 scale_g = _mm256_add_ps(_mm256_mul_ps(Lg, dot_product), _mm256_mul_ps(Ag, kd_ps));
                __m256 scale_b = _mm256_add_ps(_mm256_mul_ps(Lb, dot_product), _mm256_mul_ps(Ab, kd_ps));
                __m256 shaded_r = _mm256_mul_ps(_mm256_mul_ps(color_r, kd_ps), scale_r);
                __m256 shaded_g = _mm256_mul_ps(_mm256_mul_ps(color_g, kd_ps), scale_g);
                __m256 shaded_b = _mm256_mul_ps(_mm256_mul_ps(color_b, kd_ps), scale_b);
                __m256 final_r = _mm256_min_ps(_mm256_max_ps(shaded_r, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
                __m256 final_g = _mm256_min_ps(_mm256_max_ps(shaded_g, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
                __m256 final_b = _mm256_min_ps(_mm256_max_ps(shaded_b, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
                __m256 scale255 = _mm256_set1_ps(255.0f);
                final_r = _mm256_mul_ps(final_r, scale255);
                final_g = _mm256_mul_ps(final_g, scale255);
                final_b = _mm256_mul_ps(final_b, scale255);
                float r[8], g[8], b[8], d[8];
                _mm256_storeu_ps(r, final_r);
                _mm256_storeu_ps(g, final_g);
                _mm256_storeu_ps(b, final_b);
                _mm256_storeu_ps(d, depth);
                for (int i = 0; i < 8; i++) {
                    if (mask_int & (1 << i)) {
                        renderer.canvas.draw(x + i, y, (unsigned char)r[i], (unsigned char)g[i], (unsigned char)b[i]);
                        renderer.zbuffer(x + i, y) = d[i];
                    }
                }
        }
    }
}
public:
    // Draw triangle only inside tile [startX,endX) x [startY,endY). Uses parentMesh->kd. No mutex (tiles disjoint).
    void drawParallel(Renderer& renderer, Light& L, int startX, int startY, int endX, int endY) {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;
        if (minV.x >= endX || maxV.x < startX || minV.y >= endY || maxV.y < startY) return;

        float kd = parentMesh->kd;
        int yStart = (std::max)((int)minV.y, startY);
        int yEnd = (std::min)((int)std::ceil(maxV.y), endY);
        int xStart = (std::max)((int)minV.x, startX);
        int xEnd = (std::min)((int)std::ceil(maxV.x), endX);
        drawTileSIMD(renderer, L, kd, yStart, yEnd, xStart, xEnd, startX, endX);
    }

    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;

        int yStart = (int)(minV.y);
        int yEnd = (int)std::ceil(maxV.y);
        int xMin = (int)(minV.x);
        int xMax = (int)std::ceil(maxV.x);
        int widthLimit = (int)renderer.canvas.getWidth() - 7;
        int xEnd = std::min(xMax, widthLimit);
        int w = (int)renderer.canvas.getWidth();
        drawTileSIMD(renderer, L, kd, yStart, yEnd, xMin, xEnd, 0, w);

        for (int y = yStart; y < yEnd; y++) {
            for (int x = xEnd; x < xMax; x++) {
                float alpha, beta, gamma;
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {
                    colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                    vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                    normal.normalise();
                    if (renderer.zbuffer(x, y) > depth && depth > 0.01f) {
                        L.omega_i.normalise();
                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka);
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }

    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    void getBounds(vec2D& minV, vec2D& maxV) {
        minV = vec2D(v[0].p);
        maxV = vec2D(v[0].p);
        for (unsigned int i = 1; i < 3; i++) {
            minV.x = std::min(minV.x, v[i].p[0]);
            minV.y = std::min(minV.y, v[i].p[1]);
            maxV.x = std::max(maxV.x, v[i].p[0]);
            maxV.y = std::max(maxV.y, v[i].p[1]);
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) {
        getBounds(minV, maxV);
        minV.x = std::max(minV.x, static_cast<float>(0));
        minV.y = std::max(minV.y, static_cast<float>(0));
        maxV.x = std::min(maxV.x, static_cast<float>(canvas.getWidth()));
        maxV.y = std::min(maxV.y, static_cast<float>(canvas.getHeight()));
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas) {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        for (int y = (int)minV.y; y < (int)maxV.y; y++) {
            for (int x = (int)minV.x; x < (int)maxV.x; x++) {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() {
        for (unsigned int i = 0; i < 3; i++) {
            v[i].p.display();
        }
        std::cout << std::endl;
    }
};
