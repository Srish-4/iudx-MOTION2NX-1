// MIT License
//
// Copyright (c) 2020 Lennart Braun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstddef>
#include <vector>

namespace MOTION {

namespace tensor {
struct MaxPoolOp;
using AveragePoolOp = MaxPoolOp;
struct Conv2DOp;
struct GemmOp;
struct HammOp;
struct JoinOp;
}  // namespace tensor

template <typename T>
std::vector<T> matrix_multiply(std::size_t dim_l, std::size_t dim_m, std::size_t dim_n,
                               const std::vector<T>& A, const std::vector<T>& B);

template <typename T>
void matrix_multiply(std::size_t dim_l, std::size_t dim_m, std::size_t dim_n, const T* A,
                     const T* B, T* output);

template <typename T>
void matrix_multiply(const tensor::GemmOp&, const T* A, const T* B, T* output);

template <typename T> 
void transpose(const tensor::GemmOp&, const T* A, const T* B, T* output_A, T* output_B);


template <typename T>
void hadamard_matrix_multiply(const tensor::HammOp&, const T* A, const T* B, T* output);

template <typename T>
std::vector<T> join_matrices(std::size_t dim_l, std::size_t dim_m, std::size_t dim_n,
                               const std::vector<T>& A, const std::vector<T>& B);

template <typename T>
void join_matrices(std::size_t dim_l, std::size_t dim_m, std::size_t dim_n, const T* A,
                     const T* B, T* output);

template <typename T>
void join_matrices(const tensor::JoinOp&, const T* A, const T* B, T* output);

template <typename T>
std::vector<T> convolution(const tensor::Conv2DOp&, const std::vector<T>& input,
                           const std::vector<T>& kernel);

template <typename T>
void convolution(const tensor::Conv2DOp&, const T* input, const T* kernel, T* output);

template <typename T>
void sum_pool(const tensor::AveragePoolOp&, const T* input, T* output);

}  // namespace MOTION
