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

#include <memory>
#include <type_traits>
#include <vector>

#include "oblivious_transfer/ot_provider.h"
#include "tensor/tensor_op.h"
#include "utility/reusable_future.h"
#include "utility/type_traits.hpp"

namespace ENCRYPTO::ObliviousTransfer {
class OTProvider;
class OTProviderManager;
template <typename T>
class ACOTSender;
template <typename T>
class ACOTReceiver;
}  // namespace ENCRYPTO::ObliviousTransfer

namespace MOTION {

namespace Communication {
class CommunicationLayer;
}

class ArithmeticProvider;
class Logger;

template <typename T>
class BitIntegerMultiplicationIntSide {
 public:
  BitIntegerMultiplicationIntSide(std::size_t batch_size, std::size_t vector_size,
                                  ENCRYPTO::ObliviousTransfer::OTProvider&);
  ~BitIntegerMultiplicationIntSide();
  void set_inputs(std::vector<T>&& inputs);
  void set_inputs(const std::vector<T>& inputs);
  void set_inputs(const T* inputs);
  void compute_outputs();
  std::vector<T> get_outputs();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::size_t batch_size_;
  std::size_t vector_size_;
  std::vector<T> outputs_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
};

template <typename T>
class BitIntegerMultiplicationBitSide {
 public:
  BitIntegerMultiplicationBitSide(std::size_t batch_size, std::size_t vector_size,
                                  ENCRYPTO::ObliviousTransfer::OTProvider&);
  ~BitIntegerMultiplicationBitSide();
  void set_inputs(ENCRYPTO::BitVector<>&& inputs);
  void set_inputs(const ENCRYPTO::BitVector<>& inputs);
  void compute_outputs();
  std::vector<T> get_outputs();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::size_t batch_size_;
  std::size_t vector_size_;
  std::vector<T> outputs_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
  std::shared_ptr<Logger> logger_;
};

template <typename T>
class IntegerMultiplicationSender {
 public:
  IntegerMultiplicationSender(std::size_t batch_size, std::size_t vector_size,
                              ENCRYPTO::ObliviousTransfer::OTProvider&);
  ~IntegerMultiplicationSender();
  void set_inputs(std::vector<T>&& inputs);
  void set_inputs(const std::vector<T>& inputs);
  void set_inputs(const T* inputs);
  void compute_outputs();
  std::vector<T> get_outputs();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::size_t batch_size_;
  std::size_t vector_size_;
  std::vector<T> outputs_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
};

template <typename T>
class IntegerMultiplicationReceiver {
 public:
  IntegerMultiplicationReceiver(std::size_t batch_size, std::size_t vector_size,
                                ENCRYPTO::ObliviousTransfer::OTProvider&);
  ~IntegerMultiplicationReceiver();
  void set_inputs(std::vector<T>&& inputs);
  void set_inputs(const std::vector<T>& inputs);
  void set_inputs(const T* inputs);
  void compute_outputs();
  std::vector<T> get_outputs();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::size_t batch_size_;
  std::size_t vector_size_;
  std::vector<T> outputs_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
  std::shared_ptr<Logger> logger_;
};

template <typename T>
class MatrixMultiplicationRHS {
 public:
  MatrixMultiplicationRHS(std::size_t l, std::size_t m, std::size_t n, ArithmeticProvider&);
  ~MatrixMultiplicationRHS();
  void set_input(std::vector<T>&& inputs);
  void set_input(const std::vector<T>& inputs);
  void set_input(const T* inputs);
  void compute_output();
  std::vector<T> get_output();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::array<std::size_t, 3> dims_;
  std::vector<T> output_;
  std::unique_ptr<IntegerMultiplicationSender<T>> mult_sender_;
  bool is_output_ready_;
};

template <typename T>
class MatrixMultiplicationLHS {
 public:
  MatrixMultiplicationLHS(std::size_t l, std::size_t m, std::size_t n, ArithmeticProvider&);
  ~MatrixMultiplicationLHS();
  void set_input(std::vector<T>&& inputs);
  void set_input(const std::vector<T>& inputs);
  void set_input(const T* inputs);
  void compute_output();
  std::vector<T> get_output();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::array<std::size_t, 3> dims_;
  std::vector<T> output_;
  std::unique_ptr<IntegerMultiplicationReceiver<T>> mult_receiver_;
  std::shared_ptr<Logger> logger_;
  bool is_output_ready_;
};

template <typename T>
class HadamardMatrixMultiplicationRHS {
 public:
  HadamardMatrixMultiplicationRHS(std::size_t l, std::size_t m, ArithmeticProvider&);
  ~HadamardMatrixMultiplicationRHS();
  void set_input(std::vector<T>&& inputs);
  void set_input(const std::vector<T>& inputs);
  void set_input(const T* inputs);
  void compute_output();
  std::vector<T> get_output();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::array<std::size_t, 2> dims_;
  std::vector<T> output_;
  std::unique_ptr<IntegerMultiplicationSender<T>> mult_sender_;
  bool is_output_ready_;
};

template <typename T>
class HadamardMatrixMultiplicationLHS {
 public:
  HadamardMatrixMultiplicationLHS(std::size_t l, std::size_t m, ArithmeticProvider&);
  ~HadamardMatrixMultiplicationLHS();
  void set_input(std::vector<T>&& inputs);
  void set_input(const std::vector<T>& inputs);
  void set_input(const T* inputs);
  void compute_output();
  std::vector<T> get_output();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::array<std::size_t, 2> dims_;
  std::vector<T> output_;
  std::unique_ptr<IntegerMultiplicationReceiver<T>> mult_receiver_;
  std::shared_ptr<Logger> logger_;
  bool is_output_ready_;
};

template <typename T>
class ConvolutionInputSide {
 public:
  ConvolutionInputSide(tensor::Conv2DOp, ArithmeticProvider&);
  ~ConvolutionInputSide();
  void set_input(std::vector<T>&& inputs);
  void set_input(const std::vector<T>& inputs);
  void set_input(const T* inputs);
  void compute_output();
  std::vector<T> get_output();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  const tensor::Conv2DOp conv_op_;
  std::vector<T> output_;
  std::unique_ptr<MatrixMultiplicationRHS<T>> matrix_rhs_;
  bool is_output_ready_;
};

template <typename T>
class ConvolutionKernelSide {
 public:
  ConvolutionKernelSide(tensor::Conv2DOp conv_op, ArithmeticProvider&);
  ~ConvolutionKernelSide();
  void set_input(std::vector<T>&& inputs);
  void set_input(const std::vector<T>& inputs);
  void set_input(const T* inputs);
  void compute_output();
  std::vector<T> get_output();
  void clear() noexcept;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  const tensor::Conv2DOp conv_op_;
  std::vector<T> output_;
  std::unique_ptr<MatrixMultiplicationLHS<T>> matrix_lhs_;
  std::shared_ptr<Logger> logger_;
  bool is_output_ready_;
};

class ArithmeticProvider {
 public:
  ArithmeticProvider(ENCRYPTO::ObliviousTransfer::OTProvider&, std::shared_ptr<Logger>);

  template <typename T>
  std::unique_ptr<BitIntegerMultiplicationIntSide<T>> register_bit_integer_multiplication_int_side(
      std::size_t batch_size, std::size_t vector_size = 1);
  template <typename T>
  std::unique_ptr<BitIntegerMultiplicationBitSide<T>> register_bit_integer_multiplication_bit_side(
      std::size_t batch_size, std::size_t vector_size = 1);

  template <typename T>
  std::unique_ptr<IntegerMultiplicationSender<T>> register_integer_multiplication_send(
      std::size_t batch_size, std::size_t vector_size);
  template <typename T>
  std::unique_ptr<IntegerMultiplicationSender<T>> register_integer_multiplication_send(
      std::size_t batch_size) {
    return register_integer_multiplication_send<T>(batch_size, 1);
  }
  template <typename T>
  std::unique_ptr<IntegerMultiplicationReceiver<T>> register_integer_multiplication_receive(
      std::size_t batch_size, std::size_t vector_size);
  template <typename T>
  std::unique_ptr<IntegerMultiplicationReceiver<T>> register_integer_multiplication_receive(
      std::size_t batch_size) {
    return register_integer_multiplication_receive<T>(batch_size, 1);
  }

  template <typename T>
  std::unique_ptr<MatrixMultiplicationRHS<T>> register_matrix_multiplication_rhs(std::size_t dim_l,
                                                                                 std::size_t dim_m,
                                                                                 std::size_t dim_n);
  template <typename T>
  std::unique_ptr<MatrixMultiplicationLHS<T>> register_matrix_multiplication_lhs(std::size_t dim_l,
                                                                                 std::size_t dim_m,
                                                                                 std::size_t dim_n);

  template <typename T>
  std::unique_ptr<HadamardMatrixMultiplicationRHS<T>> register_hadamard_matrix_multiplication_rhs(std::size_t dim_l,
                                                                                                  std::size_t dim_m);
  template <typename T>
  std::unique_ptr<HadamardMatrixMultiplicationLHS<T>> register_hadamard_matrix_multiplication_lhs(std::size_t dim_l,
                                                                                                  std::size_t dim_m);


  template <typename T>
  std::unique_ptr<ConvolutionInputSide<T>> register_convolution_input_side(tensor::Conv2DOp);
  template <typename T>
  std::unique_ptr<ConvolutionKernelSide<T>> register_convolution_kernel_side(tensor::Conv2DOp);

 private:
  ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider_;
  std::shared_ptr<Logger> logger_;
};

class ArithmeticProviderManager {
 public:
  ArithmeticProviderManager(MOTION::Communication::CommunicationLayer&,
                            ENCRYPTO::ObliviousTransfer::OTProviderManager&,
                            std::shared_ptr<Logger>);
  ~ArithmeticProviderManager();

  ArithmeticProvider& get_provider(std::size_t party_id) { return *providers_.at(party_id); }

 private:
  MOTION::Communication::CommunicationLayer& comm_layer_;
  std::vector<std::unique_ptr<ArithmeticProvider>> providers_;
};

}  // namespace MOTION
