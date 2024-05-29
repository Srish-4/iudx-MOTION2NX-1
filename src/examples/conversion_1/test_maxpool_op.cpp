/*
./bin/test_maxpool_op --my-id 1 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy
--boolean-protocol yao
*/

// MIT License
//
// Copyright (c) 2021 Lennart Braun
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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <stdexcept>

#include <boost/algorithm/string.hpp>
#include <boost/json/serialize.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include "algorithm/circuit_loader.h"
#include "base/gate_factory.h"
#include "base/two_party_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "compute_server/compute_server.h"
#include "statistics/analysis.h"
#include "utility/logger.h"

#include "base/two_party_tensor_backend.h"
#include "protocols/beavy/tensor.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"

#include "utility/new_fixed_point.h"

namespace po = boost::program_options;

static std::vector<uint64_t> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
  return MOTION::Helpers::RandomVector<uint64_t>(dims.get_data_size());
}

struct Options {
  std::size_t threads;
  bool json;
  std::size_t num_repetitions;
  std::size_t num_simd;
  bool sync_between_setup_and_online;
  MOTION::MPCProtocol arithmetic_protocol;
  MOTION::MPCProtocol boolean_protocol;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
};

std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("config-file", po::value<std::string>(), "config file containing options")
    ("my-id", po::value<std::size_t>()->required(), "my party id")
    ("party", po::value<std::vector<std::string>>()->multitoken(),
     "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("threads", po::value<std::size_t>()->default_value(0), "number of threads to use for gate evaluation")
    ("json", po::bool_switch()->default_value(false), "output data in JSON format")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("sync-between-setup-and-online", po::bool_switch()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false), "just build the circuit, but not execute it")
    ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  bool help = vm["help"].as<bool>();
  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }
  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::store(po::parse_config_file(ifs, desc), vm);
  }
  try {
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error:" << e.what() << "\n\n";
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  options.my_id = vm["my-id"].as<std::size_t>();
  options.threads = vm["threads"].as<std::size_t>();
  options.json = vm["json"].as<bool>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.num_simd = vm["num-simd"].as<std::size_t>();
  options.sync_between_setup_and_online = vm["sync-between-setup-and-online"].as<bool>();
  options.no_run = vm["no-run"].as<bool>();
  if (options.my_id > 1) {
    std::cerr << "my-id must be one of 0 and 1\n";
    return std::nullopt;
  }

  auto arithmetic_protocol = vm["arithmetic-protocol"].as<std::string>();
  boost::algorithm::to_lower(arithmetic_protocol);
  if (arithmetic_protocol == "gmw") {
    options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticGMW;
  } else if (arithmetic_protocol == "beavy") {
    options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticBEAVY;
  } else {
    std::cerr << "invalid protocol: " << arithmetic_protocol << "\n";
    return std::nullopt;
  }
  auto boolean_protocol = vm["boolean-protocol"].as<std::string>();
  boost::algorithm::to_lower(boolean_protocol);
  if (boolean_protocol == "yao") {
    options.boolean_protocol = MOTION::MPCProtocol::Yao;
  } else if (boolean_protocol == "gmw") {
    options.boolean_protocol = MOTION::MPCProtocol::BooleanGMW;
  } else if (boolean_protocol == "beavy") {
    options.boolean_protocol = MOTION::MPCProtocol::BooleanBEAVY;
  } else {
    std::cerr << "invalid protocol: " << boolean_protocol << "\n";
    return std::nullopt;
  }

  const auto parse_party_argument =
      [](const auto& s) -> std::pair<std::size_t, MOTION::Communication::tcp_connection_config> {
    const static std::regex party_argument_re("([01]),([^,]+),(\\d{1,5})");
    std::smatch match;
    if (!std::regex_match(s, match, party_argument_re)) {
      throw std::invalid_argument("invalid party argument");
    }
    auto id = boost::lexical_cast<std::size_t>(match[1]);
    auto host = match[2];
    auto port = boost::lexical_cast<std::uint16_t>(match[3]);
    return {id, {host, port}};
  };

  const std::vector<std::string> party_infos = vm["party"].as<std::vector<std::string>>();
  if (party_infos.size() != 2) {
    std::cerr << "expecting two --party options\n";
    return std::nullopt;
  }

  options.tcp_config.resize(2);
  std::size_t other_id = 2;

  const auto [id0, conn_info0] = parse_party_argument(party_infos[0]);
  const auto [id1, conn_info1] = parse_party_argument(party_infos[1]);
  if (id0 == id1) {
    std::cerr << "need party arguments for party 0 and 1\n";
    return std::nullopt;
  }
  options.tcp_config[id0] = conn_info0;
  options.tcp_config[id1] = conn_info1;

  return options;
}

std::unique_ptr<MOTION::Communication::CommunicationLayer> setup_communication(
    const Options& options) {
  MOTION::Communication::TCPSetupHelper helper(options.my_id, options.tcp_config);
  return std::make_unique<MOTION::Communication::CommunicationLayer>(options.my_id,
                                                                     helper.setup_connections());
}

auto create_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  // retrieve the gate factories for the chosen protocols
  auto& arithmetic_tof = backend.get_tensor_op_factory(options.arithmetic_protocol);
  auto& yao_tof = backend.get_tensor_op_factory(options.boolean_protocol);
  auto& boolean_tof = backend.get_tensor_op_factory(MOTION::MPCProtocol::BooleanBEAVY);

  // const MOTION::tensor::GemmOp gemm_op = {
  //     .input_A_shape_ = {4, 1}, .input_B_shape_ = {1, 4}, .output_shape_ = {4, 4}};

  // added on 20th September
  // channels , rows , column
  const MOTION::tensor::MaxPoolOp maxpool_op = {.input_shape_ = {1, 3, 3},
                                                .output_shape_ = {1, 1, 1},
                                                .kernel_shape_ = {3, 3},
                                                .strides_ = {1, 1}};

  // const auto input_A_dims = gemm_op.get_input_A_tensor_dims();
  // const auto input_B_dims = gemm_op.get_input_B_tensor_dims();
  // const auto output_dims = gemm_op.get_output_tensor_dims();

  const auto input_A_dims = maxpool_op.get_input_tensor_dims();
  const auto input_B_dims = maxpool_op.get_input_tensor_dims();
  const auto output_dims = maxpool_op.get_output_tensor_dims();

  // share the inputs using the arithmetic protocol
  // NB: the inputs need to always be specified in the same order:
  // here we first specify the input of party 0, then that of party 1

  MOTION::tensor::TensorCP tensor_a, tensor_b, tensor_output;

  if (options.my_id == 0) {  // auto [input_A_promise, tensor_input_A] =
    //     arithmetic_tof.make_arithmetic_64_tensor_input_my(input_A_dims);
    auto tensor_input_B = arithmetic_tof.make_arithmetic_64_tensor_input_other(input_B_dims);
    // std::vector<uint64_t> input_A = {1, 2, 3, 4};

    // input_A_promise.set_value(input_A);

    // tensor_a = tensor_input_A;
    tensor_b = tensor_input_B;
  } else {
    // auto tensor_input_A = arithmetic_tof.make_arithmetic_64_tensor_input_other(input_A_dims);
    auto [input_B_promise, tensor_input_B] =
        arithmetic_tof.make_arithmetic_64_tensor_input_my(input_B_dims);

    auto a = MOTION::new_fixed_point::encode<std::uint64_t, float>(1, 13);
    auto b = MOTION::new_fixed_point::encode<std::uint64_t, float>(2, 13);
    auto c = MOTION::new_fixed_point::encode<std::uint64_t, float>(3, 13);
    auto d = MOTION::new_fixed_point::encode<std::uint64_t, float>(4, 13);
    auto e = MOTION::new_fixed_point::encode<std::uint64_t, float>(5, 13);
    auto f = MOTION::new_fixed_point::encode<std::uint64_t, float>(6, 13);
    auto g = MOTION::new_fixed_point::encode<std::uint64_t, float>(7, 13);
    auto h = MOTION::new_fixed_point::encode<std::uint64_t, float>(8, 13);
    auto i = MOTION::new_fixed_point::encode<std::uint64_t, float>(9, 13);
    // auto a2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-1.5, 13);
    // auto b2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-2.5, 13);
    // auto c2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-3, 13);
    // auto d2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-4.5, 13);
    // auto e2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-5.5, 13);
    // auto f2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-6, 13);
    // auto g2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-7, 13);
    // auto h2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-8, 13);
    // auto i2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(-9, 13);

    // auto i = MOTION::new_fixed_point::encode<std::uint64_t, float>(3.5, 13);

    // std::vector<uint64_t> input_B = {a, b, c, d, e, f, g, h, i, a2, b2, c2, d2, e2, f2, g2, h2,
    // i2};
    std::vector<uint64_t> input_B = {a, b, c, d, e, f, g, h, i};
    input_B_promise.set_value(input_B);

    // tensor_a = tensor_input_A;
    tensor_b = tensor_input_B;
  }

  // added on 20th September
  // booleanBEAVY
  // std::function<MOTION::tensor::TensorCP(const MOTION::tensor::TensorCP&)> make_maxPool;
  // make_maxPool = [&](const auto& input) {
  //   const auto yao_tensor = yao_tof.make_tensor_conversion(MOTION::MPCProtocol::Yao, input);
  //   const auto boolean_tensor =
  //       yao_tof.make_tensor_conversion(MOTION::MPCProtocol::BooleanBEAVY, yao_tensor);
  //   const auto maxPool_tensor = boolean_tof.make_tensor_maxpool_op(maxpool_op, boolean_tensor);
  //   return boolean_tof.make_tensor_conversion(options.arithmetic_protocol, maxPool_tensor);
  // };

  // YAO
  std::function<MOTION::tensor::TensorCP(const MOTION::tensor::TensorCP&)> make_maxPool;
  make_maxPool = [&](const auto& input) {
    const auto yao_tensor = yao_tof.make_tensor_conversion(MOTION::MPCProtocol::Yao, input);
    const auto maxPool_tensor = yao_tof.make_tensor_maxpool_op(maxpool_op, yao_tensor);
    return yao_tof.make_tensor_conversion(options.arithmetic_protocol, maxPool_tensor);
  };

  auto output = make_maxPool(tensor_b);

  ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> output_future;
  if (options.my_id == 0) {
    arithmetic_tof.make_arithmetic_tensor_output_other(output);
  } else {
    output_future = arithmetic_tof.make_arithmetic_64_tensor_output_my(output);
  }
  return output_future;
}

void run_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto output_future = create_composite_circuit(options, backend);
  backend.run();
  if (options.my_id == 1) {
    auto interm = output_future.get();
    std::cout << "The result is:\n[";
    for (int i = 0; i < interm.size(); ++i) {
      std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(interm[i], 13) << " ";
    }
    std::cout << "]" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  auto options = parse_program_options(argc, argv);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  try {
    auto comm_layer = setup_communication(*options);
    auto logger = std::make_shared<MOTION::Logger>(options->my_id,
                                                   boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);
    MOTION::TwoPartyTensorBackend backend(*comm_layer, options->threads,
                                          options->sync_between_setup_and_online, logger);
    run_composite_circuit(*options, backend);
    comm_layer->shutdown();
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
