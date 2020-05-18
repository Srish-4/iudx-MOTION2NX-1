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

#include "conversion.h"

#include <cstdint>

#include "protocols/gmw/wire.h"
#include "wire.h"
#include "yao_provider.h"

namespace MOTION::proto::yao {

YaoToBooleanGMWGateGarbler::YaoToBooleanGMWGateGarbler(std::size_t gate_id, YaoProvider&,
                                                       YaoWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires, [num_simd] {
    auto wire = std::make_shared<gmw::BooleanGMWWire>(num_simd);
    wire->get_share().Resize(num_simd);
    return wire;
  });
}

void YaoToBooleanGMWGateGarbler::evaluate_setup() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_yao = inputs_[wire_i];
    auto& wire_gmw = outputs_[wire_i];
    wire_yao->wait_setup();
    const auto& keys = wire_yao->get_keys();
    auto& share = wire_gmw->get_share();
    assert(share.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      share.Set(bool(*keys[simd_j].data() & std::byte(0x01)), simd_j);
    }
    outputs_[wire_i]->set_online_ready();
  }
}

void YaoToBooleanGMWGateGarbler::evaluate_online() {
  // nothing to do
}

YaoToBooleanGMWGateEvaluator::YaoToBooleanGMWGateEvaluator(std::size_t gate_id, YaoProvider&,
                                                           YaoWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires, [num_simd] {
    auto wire = std::make_shared<gmw::BooleanGMWWire>(num_simd);
    wire->get_share().Resize(num_simd);
    return wire;
  });
}

void YaoToBooleanGMWGateEvaluator::evaluate_setup() {
  // nothing to do
}

void YaoToBooleanGMWGateEvaluator::evaluate_online() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_yao = inputs_[wire_i];
    auto& wire_gmw = outputs_[wire_i];
    wire_yao->wait_online();
    const auto& keys = wire_yao->get_keys();
    auto& share = wire_gmw->get_share();
    assert(share.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      share.Set(bool(*keys[simd_j].data() & std::byte(0x01)), simd_j);
    }
    outputs_[wire_i]->set_online_ready();
  }
}

}  // namespace MOTION::proto::yao