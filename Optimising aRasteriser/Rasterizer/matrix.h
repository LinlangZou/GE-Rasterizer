#pragma once
#include <iostream>
#include <vector>
#include <immintrin.h>
#include "vec4.h"

// Matrix class for 4x4 transformation matrices
class alignas(16) matrix {
    union {
        float m[4][4]; // 2D array representation of the matrix
        float a[16];   // 1D array representation of the matrix for linear access
        __m128 row[4]; // store row of matrix by 4 SIMD vectors
    };

public:
    // Default constructor initializes the matrix as an identity matrix
    matrix() {
        identity();
    }

    // Access matrix elements by row and column
    float& operator()(unsigned int row, unsigned int col) { return m[row][col]; }

    // Display the matrix elements in a readable format
    void display() {
        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++)
                std::cout << m[i][j] << '\t';
            std::cout << std::endl;
        }
    }

    // Multiply the matrix by a 4D vector
    // Input Variables:
    // - v: vec4 object to multiply with the matrix
    // Returns the resulting transformed vec4
    vec4 operator * (const vec4& v) const {
        vec4 result;
        __m128 vec = _mm_loadu_ps(v.getV());
        __m128 row0 = _mm_loadu_ps(&a[0]);
        __m128 row1 = _mm_loadu_ps(&a[4]);
        __m128 row2 = _mm_loadu_ps(&a[8]);
        __m128 row3 = _mm_loadu_ps(&a[12]);
        __m128 res0 = _mm_dp_ps(row0, vec, 0xF1);
        __m128 res1 = _mm_dp_ps(row1, vec, 0xF1);
        __m128 res2 = _mm_dp_ps(row2, vec, 0xF1);
        __m128 res3 = _mm_dp_ps(row3, vec, 0xF1);
        result[0] = _mm_cvtss_f32(res0);
        result[1] = _mm_cvtss_f32(res1);
        result[2] = _mm_cvtss_f32(res2);
        result[3] = _mm_cvtss_f32(res3);
        return result;
    }

    // Multiply the matrix by another matrix
    // Input Variables:
    // - mx: Another matrix to multiply with
    // Returns the resulting matrix
    matrix operator * (const matrix& mx) const {
        matrix ret;
        alignas(16) float row_values[4];
        for (int i = 0; i < 4; i++) {
            _mm_store_ps(row_values, row[i]);
            ret.row[i] = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(_mm_set1_ps(row_values[0]), mx.row[0]),
                    _mm_mul_ps(_mm_set1_ps(row_values[1]), mx.row[1])
                ),
                _mm_add_ps(
                    _mm_mul_ps(_mm_set1_ps(row_values[2]), mx.row[2]),
                    _mm_mul_ps(_mm_set1_ps(row_values[3]), mx.row[3])
                )
            );
        }
        return ret;
    }

    // Create a perspective projection matrix
    // Input Variables:
    // - fov: Field of view in radians
    // - aspect: Aspect ratio of the viewport
    // - n: Near clipping plane
    // - f: Far clipping plane
    // Returns the perspective matrix
    static matrix makePerspective(float fov, float aspect, float n, float f) {
        matrix m;
        m.zero();
        float tanHalfFov = std::tan(fov / 2.0f);

        m.a[0] = 1.0f / (aspect * tanHalfFov);
        m.a[5] = 1.0f / tanHalfFov;
        m.a[10] = -f / (f - n);
        m.a[11] = -(f * n) / (f - n);
        m.a[14] = -1.0f;
        return m;
    }

    // Create a translation matrix
    // Input Variables:
    // - tx, ty, tz: Translation amounts along the X, Y, and Z axes
    // Returns the translation matrix
    static matrix makeTranslation(float tx, float ty, float tz) {
        matrix m;
        m.identity();
        m.a[3] = tx;
        m.a[7] = ty;
        m.a[11] = tz;
        return m;
    }

    // Create a rotation matrix around the Z-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static matrix makeRotateZ(float aRad) {
        matrix m;
        m.identity();
        m.a[0] = std::cos(aRad);
        m.a[1] = -std::sin(aRad);
        m.a[4] = std::sin(aRad);
        m.a[5] = std::cos(aRad);
        return m;
    }

    // Create a rotation matrix around the X-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static matrix makeRotateX(float aRad) {
        matrix m;
        m.identity();
        m.a[5] = std::cos(aRad);
        m.a[6] = -std::sin(aRad);
        m.a[9] = std::sin(aRad);
        m.a[10] = std::cos(aRad);
        return m;
    }

    // Create a rotation matrix around the Y-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static matrix makeRotateY(float aRad) {
        matrix m;
        m.identity();
        m.a[0] = std::cos(aRad);
        m.a[2] = std::sin(aRad);
        m.a[8] = -std::sin(aRad);
        m.a[10] = std::cos(aRad);
        return m;
    }

    // Create a composite rotation matrix from X, Y, and Z rotations
    // Input Variables:
    // - x, y, z: Rotation angles in radians around each axis
    // Returns the composite rotation matrix
    static matrix makeRotateXYZ(float x, float y, float z) {
        return matrix::makeRotateX(x) * matrix::makeRotateY(y) * matrix::makeRotateZ(z);
    }

    // Create a scaling matrix
    // Input Variables:
    // - s: Scaling factor
    // Returns the scaling matrix
    static matrix makeScale(float s) {
        matrix m;
        s = std::max(s, 0.01f); // Ensure scaling factor is not too small
        m.identity();
        m.a[0] = s;
        m.a[5] = s;
        m.a[10] = s;
        return m;
    }

    // Create an identity matrix
    // Returns an identity matrix
    static matrix makeIdentity() {
        matrix m;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m.m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return m;
    }

private:
    // Set all elements of the matrix to 0
    void zero() {
        for (unsigned int i = 0; i < 16; i++)
            a[i] = 0.f;
    }

    // Set the matrix as an identity matrix
    void identity() {
        __m128 zero = _mm_setzero_ps();
        _mm_storeu_ps(&a[0], zero);
        _mm_storeu_ps(&a[4], zero);
        _mm_storeu_ps(&a[8], zero);
        _mm_storeu_ps(&a[12], zero);
        a[0] = 1.0f;
        a[5] = 1.0f;
        a[10] = 1.0f;
        a[15] = 1.0f;
    }
};


