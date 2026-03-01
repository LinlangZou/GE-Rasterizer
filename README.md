# Optimising Rasteriser

A high-performance software rasteriser in C++, optimised with SIMD (SSE4.1 / AVX2), vertex caching, and tile-based multithreading. Built for the Games Engineering module at the University of Warwick.

---

## Features

| Optimisation | Description |
|-------------|-------------|
| **Matrix & vector SIMD** | 4×4 matrix × vector and matrix × matrix use SSE4.1 (`_mm_dp_ps`, aligned loads) in `matrix.h`; `vec4` is 16-byte aligned for fast loads. |
| **Vertex cache** | Each vertex is transformed once per frame per mesh and reused for all triangles that share it (no duplicate transforms). |
| **Triangle rasterisation SIMD** | AVX2 path in `triangle.h` processes 8 pixels at a time (`__m256`) for barycentrics, depth, and shading in the inner loop. |
| **Tile-based multithreading** | Screen split into 256×256 tiles; a thread pool processes one tile per job; triangles are binned to overlapping tiles. No per-pixel locks. |

Output is **bit-identical** to the original: same geometry, lighting, and image for the same inputs.

---

## Tech Stack

- **Language:** C++
- **SIMD:** SSE4.1 (matrix/vector), AVX2 (triangle inner loop)
- **Concurrency:** `std::thread` via a custom `ThreadPool` (batch submit + `waitIdle()`)
- **Build:** Visual Studio (see `Rasterizer.sln`)

---

## Project Structure

```
Optimising-Rasteriser/
├── Rasterizer/
│   ├── raster.cpp          # Main loop, vertex cache, tile binning, scene setup
│   ├── matrix.h            # 4×4 matrix with SSE4.1 (matrix×vec4, matrix×matrix)
│   ├── vec4.h              # 4D vector (alignas(16))
│   ├── triangle.h          # Triangle rasterisation (scalar + AVX2 8-wide)
│   ├── mesh.h              # Mesh, spheres, triangles
│   ├── ThreadPool.h        # Thread pool: submitBatch(), waitIdle()
│   ├── renderer.h / zbuffer.h / colour.h / light.h
│   └── ...
└── README.md
```

---

## Build & Run

1. Open `Rasterizer.sln` in Visual Studio.
2. Build in **Release** for timings (Debug is unoptimised).
3. Run the executable; resolution is 1024×768 by default (`renderer.h`).

Thread count is set to **11** in `raster.cpp` (`MT_POOL_SIZE`); change and recompile to match your CPU.

---

## Scenes & Performance

Three scenes are used for benchmarking:

- **Scene 1:** 40 cubes in two columns with random rotations.
- **Scene 2:** 6×8 grid of cubes plus a moving sphere.
- **Scene 3:** Six spheres orbiting in the XY plane plus a 10×10 grid of static spheres (custom scene for mixed workload).

Timings are reported as **segment duration (ms)** — one segment per full animation cycle (e.g. camera back-and-forth, or two orbits in Scene 3). **Speedup** is baseline time (Stage 0) divided by time at each stage.

| Stage | Added optimisation |
|-------|---------------------|
| 0 | Original (baseline) |
| 1 | + Matrix & vector SIMD |
| 2 | + Vertex transform reuse (vertex cache) |
| 3 | + Rasterisation SIMD (triangle 8-wide) |
| 4 | + Tile-based multithreading |

All three scenes show clear speedup across these stages. 
<img width="312" height="255" alt="speedup" src="https://github.com/user-attachments/assets/45906949-5969-47d9-8486-7880135f3bd3" />

---

## Author

**Linlang Zou**  
University of Warwick — Games Engineering (February 2026)

---

