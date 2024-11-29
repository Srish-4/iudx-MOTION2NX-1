/*
This code reads shares from build_debwithrelinfo/ShareFiles/A.txt and
build_debwithrelinfo/ShareFiles/B.txt and performs a ReLU operation on them.
If this code is run with a function to write output shares (in tensor_op.cpp)
output shares of this will be written. The following instructions run this code.

At the argument "--filepath " give the path of the file containing shares from build_deb.... folder
Server-0
./bin/test_tensor_maxpool --my-id 0 --party 0,::1,7002 --party 1,::1,7000 --arithmetic-protocol
beavy
--boolean-protocol yao --fractional-bits 13 --filepath cnn_outputshare --current-path
${BASE_DIR}/build_debwithrelinfo_gcc --strides 1 --pool-size 1

Server-1
./bin/test_tensor_maxpool --my-id 1 --party 0,::1,7002 --party 1,::1,7000 --arithmetic-protocol
beavy
--boolean-protocol yao --fractional-bits 13 --filepath cnn_outputshare --current-path
${BASE_DIR}/build_debwithrelinfo_gcc --strides 1 --pool-size 1
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
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <stdexcept>
#include <thread>

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

void testMemoryOccupied(bool WriteToFiles, int my_id, std::string path) {
  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();

  long page_size_kb =
      sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;
  std::cout << "RSS - " << rss << " kB\n";
  double shared_mem = share * page_size_kb;
  std::cout << "Shared Memory - " << shared_mem << " kB\n";
  std::cout << "Private Memory - " << rss - shared_mem << "kB\n";
  std::cout << std::endl;
  if (WriteToFiles == 1) {
    /////// Generate path for the AverageMemoryDetails file and MemoryDetails file
    std::string t1 = path + "/" + "AverageMemoryDetails" + std::to_string(my_id);
    std::string t2 = path + "/" + "MemoryDetails" + std::to_string(my_id);

    ///// Write to the AverageMemoryDetails files
    std::ofstream file1;
    file1.open(t1, std::ios_base::app);
    file1 << rss;
    file1 << "\n";
    file1.close();

    std::ofstream file2;
    file2.open(t2, std::ios_base::app);
    file2 << "RelU : \n";
    file2 << "RSS - " << rss << " kB\n";
    file2 << "Shared Memory - " << shared_mem << " kB\n";
    file2 << "Private Memory - " << rss - shared_mem << "kB\n";
    file2.close();
  }
}

namespace po = boost::program_options;

static std::vector<uint64_t> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
  return MOTION::Helpers::RandomVector<uint64_t>(dims.get_data_size());
}

struct Matrix {
  std::vector<uint64_t> Delta;
  std::vector<uint64_t> delta;
  int row;
  int col;
};

struct Options {
  std::size_t threads;
  bool json;
  std::size_t num_repetitions;
  std::size_t num_simd;
  bool sync_between_setup_and_online;
  MOTION::MPCProtocol arithmetic_protocol;
  MOTION::MPCProtocol boolean_protocol;
  //////////////////////////changes////////////////////////////
  Matrix input;
  std::uint64_t input_chnl;
  std::uint64_t input_rows;
  std::uint64_t input_cols;
  std::string currentpath;
  //////////////////////////////////////////////////////////////
  std::size_t fractional_bits;
  std::size_t my_id;
  std::string filepath_frombuild;
  std::uint64_t num_elements;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
  std::uint64_t pool_size, strides;
  Matrix W_file, B_file;
};

bool is_empty(std::ifstream& file) { return file.peek() == std::ifstream::traits_type::eof(); }

//////////////////New functions////////////////////////////////////////
/// In read_file also include file not there error and file empty alerts
std::uint64_t read_file(std::ifstream& indata) {
  std::string str;
  char num;
  while (indata >> std::noskipws >> num) {
    if (num != ' ' && num != '\n') {
      str.push_back(num);
    } else {
      break;
    }
  }
  std::string::size_type sz = 0;
  std::uint64_t ret = (uint64_t)std::stoull(str, &sz, 0);
  return ret;
}

std::string read_filepath(std::ifstream& indata) {
  std::string str;

  char num;
  while (indata) {
    std::getline(indata, str);
  }
  // std::cout << str << std::endl;
  return str;
}

int input_shares(Options* options, std::string p) {
  // std::ifstream indata1;

  // try {
  //   indata1.open(p);
  //   if (indata1) {
  //     std::cout << "File found\n";
  //   } else {
  //     std::cout << "File not found\n";
  //   }
  // } catch (std::ifstream::failure e) {
  //   std::cerr << "Error while opening the input share file.\n";
  //   return EXIT_FAILURE;
  // }

  // try {
  //   if (is_empty(indata1)) {
  //     // file is empty
  //   }
  // } catch (std::ifstream::failure e) {
  //   std::cerr << "Image share input file is empty.\n";
  //   return EXIT_FAILURE;
  // }

  // int input_chnl, input_rows, input_cols;
  // try {
  //   input_chnl = read_file(indata1);
  //   input_rows = read_file(indata1);
  //   input_cols = read_file(indata1);
  // } catch (std::ifstream::failure e) {
  //   std::cerr << "Error while reading columns from image shares.\n";
  //   return EXIT_FAILURE;
  // }

  // options->input_chnl = input_chnl;
  // options->input_rows = input_rows;
  // options->input_cols = input_cols;

  // int num_elements = input_chnl * input_cols * input_rows;
  // options->num_elements = num_elements;
  // std::cout << options->num_elements << "\n";

  // for (int i = 0; i < options->num_elements; ++i) {
  //   try {
  //     uint64_t m1 = read_file(indata1);
  //     options->input.Delta.push_back(m1);
  //     uint64_t m2 = read_file(indata1);
  //     options->input.delta.push_back(m2);
  //     std::cout << m1 << " " << m2 << "\n";
  //   } catch (std::ifstream::failure e) {
  //     std::cerr << "Error while reading columns from image shares.\n";
  //     return EXIT_FAILURE;
  //   }
  // }
  // indata1.close();
  options->input.Delta = {
      15130840057080310500, 2732311630999460628,  5999265575444322724,  17354190036302678058,
      17509206757624434773, 12757302977252562970, 467848558594841556,   18354156824625577761,
      11369926242703930852, 13180788076397517719, 6069004501101385937,  8736108189683060374,
      1830593949177511447,  8259731252535636521,  7745443822406729188,  15130840057080310500,
      2732311630999460628,  5999265575444322724,  17354190036302678058, 17509206757624434773,
      12757302977252562970, 467848558594841556,   18354156824625577761, 11369926242703930852,
      13180788076397517719, 6069004501101385937,  8736108189683060374,  1830593949177511447,
      8259731252535636521,  7745443822406729188};
  options->W_file.Delta = {15130840057080310500, 467848558594841556,   15130840057080310500,
                           467848558594841556,   15130840057080310500, 467848558594841556,
                           15130840057080310500, 467848558594841556};

  options->B_file.Delta = {467848558594841556};

  if (options->my_id == 0) {
    options->input.delta = {
        6859613470069005590,  6766853617053379389,  9385385440214582262,  11046059727528296421,
        10363471687557326202, 4096403903189193844,  3451458718746265459,  5643009273266552008,
        278182501930048131,   7256601518208340437,  16872412304666220063, 5884353682621150764,
        15218774696715124927, 9919842306516522705,  15888773582626106351, 6859613470069005590,
        6766853617053379389,  9385385440214582262,  11046059727528296421, 10363471687557326202,
        4096403903189193844,  3451458718746265459,  5643009273266552008,  278182501930048131,
        7256601518208340437,  16872412304666220063, 5884353682621150764,  15218774696715124927,
        9919842306516522705,  15888773582626106351};

    options->W_file.delta = {6859613470069005590, 3451458718746265459, 6859613470069005590,
                             3451458718746265459, 6859613470069005590, 3451458718746265459,
                             6859613470069005590, 3451458718746265459};

    options->B_file.delta = {3451458718746265459};
  } else {
    options->input.delta = {
        8271226587011296718,  14412202087655616471, 15060624208939267502, 6308130308774348869,
        7145735070067067611,  8660899074063319974,  15463133913558127713, 12711147551359025753,
        11091743740773882721, 5924186558189185474,  7643336270144733874,  2851754507061934186,
        5058563326171970904,  16786633019728706392, 10303414313490223605, 8271226587011296718,
        14412202087655616471, 15060624208939267502, 6308130308774348869,  7145735070067067611,
        8660899074063319974,  15463133913558127713, 12711147551359025753, 11091743740773882721,
        5924186558189185474,  7643336270144733874,  2851754507061934186,  5058563326171970904,
        16786633019728706392, 10303414313490223605};

    options->W_file.delta = {8271226587011296718,  15463133913558127713, 8271226587011296718,
                             15463133913558127713, 8271226587011296718,  15463133913558127713,
                             8271226587011296718,  15463133913558127713};

    options->B_file.delta = {15463133913558127713};
  }
}

int file_read(Options* options) {
  // std::string path = ../build_debwithrelinfo_gcc
  // std::string t1 = ../build_debwithrelinfo_gcc/server_{id}/cnn_outputshare_{id}
  std::string path = options->currentpath;
  std::string t1 = path + "/server" + std::to_string(options->my_id) + "/" +
                   options->filepath_frombuild + "_" + std::to_string(options->my_id);
  std::cout << t1 << "\n";

  std::ifstream file1;
  try {
    file1.open(t1);
    if (file1) {
      std::cout << "File found\n";
    } else {
      std::cout << "File not found\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening config_model file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(file1)) {
      // file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "config_file_model is empty.\n";
    return EXIT_FAILURE;
  }
  input_shares(options, t1);
}

//////////////////////////////////////////////////////////////////////
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
    ("fractional-bits", po::value<std::size_t>()->default_value(16),
     "number of fractional bits for fixed-point arithmetic")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ("filepath", po::value<std::string>()->required(), "Path of the shares file from build_debwithrelinfo folder")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("current-path",po::value<std::string>()->required(), "current path build_debwithrelinfo")
    ("pool-size", po::value<std::size_t>()->required(), "pool size")
    ("strides", po::value<std::size_t>()->required(), "strides")
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
  options.pool_size = vm["pool-size"].as<std::size_t>();
  options.strides = vm["strides"].as<std::size_t>();
  options.threads = vm["threads"].as<std::size_t>();
  options.json = vm["json"].as<bool>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.num_simd = vm["num-simd"].as<std::size_t>();
  options.sync_between_setup_and_online = vm["sync-between-setup-and-online"].as<bool>();
  options.no_run = vm["no-run"].as<bool>();
  //////////////////////////////////////////////////////////////////
  options.filepath_frombuild = vm["filepath"].as<std::string>();
  options.currentpath = vm["current-path"].as<std::string>();
  /////////////////////////////////////////////////////////////////
  options.fractional_bits = vm["fractional-bits"].as<std::size_t>();
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

  //////////////////////////////////////////////////////////////////
  file_read(&options);
  ////////////////////////////////////////////////////////////////////

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

void print_stats(const Options& options,
                 const MOTION::Statistics::AccumulatedRunTimeStats& run_time_stats,
                 const MOTION::Statistics::AccumulatedCommunicationStats& comm_stats) {
  if (options.json) {
    auto obj = MOTION::Statistics::to_json("tensor_gt_relu", run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("arithmetic_protocol", MOTION::ToString(options.arithmetic_protocol));
    obj.emplace("boolean_protocol", MOTION::ToString(options.boolean_protocol));
    obj.emplace("simd", options.num_simd);
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    // std::cout << obj << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats("tensor_gt_relu", run_time_stats, comm_stats);
  }
}

auto create_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  // retrieve the gate factories for the chosen protocols
  auto& arithmetic_tof = backend.get_tensor_op_factory(options.arithmetic_protocol);
  auto& yao_tof = backend.get_tensor_op_factory(options.boolean_protocol);
  auto& boolean_tof = backend.get_tensor_op_factory(MOTION::MPCProtocol::BooleanBEAVY);

  MOTION::tensor::TensorDimensions tensor_dims;
  tensor_dims.batch_size_ = 1;
  tensor_dims.num_channels_ = 2;
  tensor_dims.height_ = 5;
  tensor_dims.width_ = 3;

  const MOTION::tensor::TensorDimensions conv_weights_dims{
      .batch_size_ = 1, .num_channels_ = 2, .height_ = 2, .width_ = 2};
  const MOTION::tensor::TensorDimensions CBias1_dims{1, 1, 1, 1};

  const MOTION::tensor::Conv2DOp conv_op = {.kernel_shape_ = {1, 2, 2, 2},
                                            .input_shape_ = {2, 5, 3},
                                            .output_shape_ = {1, 4, 2},
                                            .dilations_ = {1, 1},
                                            .pads_ = {0, 0, 0, 0},
                                            .strides_ = {1, 1}};

  MOTION::tensor::TensorCP tensor_X, tensor_CW1, tensor_CB1;

  /////////////////////////////////////////////////////////////////////////
  MOTION::tensor::TensorCP tensor_input;
  auto pair_input = arithmetic_tof.make_arithmetic_64_tensor_input_shares(tensor_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_vector =
      std::move(pair_input.first);
  tensor_input = pair_input.second;
  assert(tensor_input);

  auto pairCW1 = arithmetic_tof.make_arithmetic_64_tensor_input_shares(conv_weights_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_CW1 =
      std::move(pairCW1.first);
  tensor_CW1 = pairCW1.second;

  auto pairCB1 = arithmetic_tof.make_arithmetic_64_tensor_input_shares(CBias1_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_CB1 =
      std::move(pairCB1.first);
  tensor_CB1 = pairCB1.second;

  ///////////////////////////////////////////////////////////////
  input_vector[0].set_value(options.input.Delta);
  input_vector[1].set_value(options.input.delta);
  ///////////////////////////////////////////////////////////////////

  input_promises_CW1[0].set_value(options.W_file.Delta);
  input_promises_CW1[1].set_value(options.W_file.delta);

  input_promises_CB1[0].set_value(
      options.B_file.Delta);  // Dummy Bias values (5, 1, 1, 1) for mnist
  input_promises_CB1[1].set_value(options.B_file.delta);

  int output_chnl = 1;
  int output_rows = (4 - 2 + 1) / 1;
  int output_cols = (2 - 2 + 1) / 1;

  const MOTION::tensor::MaxPoolOp maxpool_op = {
      .input_shape_ = {1, 4, 2},
      .output_shape_ = {1, output_rows, output_cols},
      .kernel_shape_ = {options.pool_size, options.pool_size},
      .strides_ = {options.strides, options.strides}};

  /////////////////////////////////////////////////////////////////////////

  // Convolution Operation
  auto conv_output = arithmetic_tof.make_tensor_conv2d_op(conv_op, tensor_input, tensor_CW1,
                                                          tensor_CB1, options.fractional_bits);

  ////////////////////////////////////////////////////////////////////////////

  std::function<MOTION::tensor::TensorCP(const MOTION::tensor::TensorCP&)> make_activation;
  make_activation = [&](const auto& input) {
    const auto negated_tensor = arithmetic_tof.make_tensor_negate(input);
    const auto boolean_tensor =
        yao_tof.make_tensor_conversion(MOTION::MPCProtocol::Yao, negated_tensor);
    const auto relu_tensor = yao_tof.make_tensor_relu_op(boolean_tensor);
    const auto finBoolean_tensor =
        yao_tof.make_tensor_conversion(options.arithmetic_protocol, relu_tensor);
    return arithmetic_tof.make_tensor_negate(finBoolean_tensor);
  };

  auto relu_output = make_activation(conv_output);

  ////////////////////////////////////////////////////////////////////////////////////

  std::function<MOTION::tensor::TensorCP(const MOTION::tensor::TensorCP&)> make_maxPool;
  make_maxPool = [&](const auto& input) {
    const auto yao_tensor = yao_tof.make_tensor_conversion(MOTION::MPCProtocol::Yao, input);
    const auto maxPool_tensor = yao_tof.make_tensor_maxpool_op(maxpool_op, yao_tensor);
    return yao_tof.make_tensor_conversion(options.arithmetic_protocol, maxPool_tensor);
  };

  auto max_output = make_maxPool(relu_output);

  ///////////////////////////////////////////////////////////////////////////////////////

  ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> output_future, main_output_future,
      main_output;

  if (options.my_id == 0) {
    arithmetic_tof.make_arithmetic_tensor_output_other(max_output);
  } else {
    main_output_future = arithmetic_tof.make_arithmetic_64_tensor_output_my(max_output);
  }

  return std::move(main_output_future);
}

void run_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto output_future = create_composite_circuit(options, backend);
  backend.run();
  if (options.my_id == 1) {
    auto main = output_future.get();

    for (int i = 0; i < main.size(); ++i) {
      long double temp =
          MOTION::new_fixed_point::decode<uint64_t, long double>(main[i], options.fractional_bits);

      std::cout << temp << " , ";
    }
  }
}
int main(int argc, char* argv[]) {
  // testMemoryOccupied();
  // std::cout << "Inside main";
  bool WriteToFiles = 1;
  auto options = parse_program_options(argc, argv);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  try {
    auto comm_layer = setup_communication(*options);
    auto logger = std::make_shared<MOTION::Logger>(options->my_id,
                                                   boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);

    MOTION::Statistics::AccumulatedRunTimeStats run_time_stats;
    MOTION::Statistics::AccumulatedCommunicationStats comm_stats;
    MOTION::TwoPartyTensorBackend backend(*comm_layer, options->threads,
                                          options->sync_between_setup_and_online, logger);
    run_composite_circuit(*options, backend);
    comm_layer->sync();
    comm_stats.add(comm_layer->get_transport_statistics());
    comm_layer->reset_transport_statistics();
    run_time_stats.add(backend.get_run_time_stats());
    testMemoryOccupied(WriteToFiles, options->my_id, options->currentpath);
    comm_layer->shutdown();
    print_stats(*options, run_time_stats, comm_stats);
    if (WriteToFiles == 1) {
      /////// Generate path for the AverageTimeDetails file and MemoryDetails file
      // std::string path = std::filesystem::current_path();
      std::string path = options->currentpath;
      std::string t1 = path + "/" + "AverageTimeDetails" + std::to_string(options->my_id);
      std::string t2 = path + "/" + "MemoryDetails" + std::to_string(options->my_id);

      ///// Write to the AverageMemoryDetails files
      std::ofstream file2;
      file2.open(t2, std::ios_base::app);
      std::string time_str =
          MOTION::Statistics::print_stats_short("tensor_gt_relu", run_time_stats, comm_stats);
      // std::cout << "Execution time string:" << time_str << "\n";
      double exec_time = std::stod(time_str);
      std::cout << "Execution time:" << exec_time << "\n";
      file2 << "Execution time - " << exec_time << "msec\n";
      file2.close();

      std::ofstream file1;
      file1.open(t1, std::ios_base::app);
      file1 << exec_time;
      file1 << "\n";
      file1.close();
    }
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    std::cerr << "ERROR Caught !!" << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}