/*
inputpath changed to imageprovider
imageprovider is file name inside server 0 and server 1 respectively
eg. (ip1_0 , ip1_1) , (ip2_0,ip2_1) and so on..
from bash file we are giving ip1 and in this file it is appended to ip1_0 and ip1_1

At the argument "--filepath " give the path of the file containing shares from build_deb.... folder
Server-0
./bin/conv_h_split --my-id 0 --party 0,::1,7002 --party 1,::1,7000 --arithmetic-protocol beavy
--boolean-protocol yao --fractional-bits 13 --config-file-model file_config_model0 --layer-id 1
--row_start 1 --row_end 6 --kernel_start 1 --kernel_end 1 --current-path
${BASE_DIR}/build_debwithrelinfo_gcc --h-actual-splits $actual_splits --h-split-index $i

Server-1
./bin/tensor_gt_mul_test --my-id 1 --party 0,::1,7002 --party 1,::1,7000 --arithmetic-protocol beavy
--boolean-protocol yao --fractional-bits 13 --config-file-model file_config_model1 --layer-id 1
--current-path ${BASE_DIR}/build_debwithrelinfo_gcc --h-actual-splits $actual_splits --h-split-index
$i

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

namespace po = boost::program_options;
int j = 0;

static std::vector<uint64_t> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
  return MOTION::Helpers::RandomVector<uint64_t>(dims.get_data_size());
}

bool is_empty(std::ifstream& file) { return file.peek() == std::ifstream::traits_type::eof(); }

void testMemoryOccupied(int WriteToFiles, int my_id, std::string path) {
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
    file2 << "Multiplication layer : \n";
    file2 << "RSS - " << rss << " kB\n";
    file2 << "Shared Memory - " << shared_mem << " kB\n";
    file2 << "Private Memory - " << rss - shared_mem << "kB\n";
    file2.close();
  }
}

//////////////////New functions////////////////////////////////////////
/// In read_file also include file not there error and file empty alerts
std::uint64_t read_file(std::ifstream& pro) {
  std::string str;
  char num;
  while (pro >> std::noskipws >> num) {
    if (num != ' ' && num != '\n') {
      str.push_back(num);
    } else {
      break;
    }
  }
  std::string::size_type sz = 0;
  std::uint64_t ret = (uint64_t)std::stoull(str, &sz, 0);
  // std::cout << "in read_file in cnn : " << ret << std::endl;
  return ret;
}

struct Matrix {
  std::vector<uint64_t> Delta;
  std::vector<uint64_t> delta;
  int row;
  int col;
  int ker;
  int chnl;
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
  int num_elements;
  //////////////////////////////////////////////////////////////
  std::size_t fractional_bits;
  std::string imageprovider;
  std::string modelpath;
  std::size_t layer_id;
  std::string currentpath;
  std::size_t my_id;
  int kernel_start;
  int kernel_end;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
  Matrix image_file;
  // Adding new variable partial_image_file to read the image shares required to compute the partial
  // output coorespodning to the horizontal split
  Matrix partial_image_file;
  Matrix W_file;
  Matrix B_file;
  /*
  Also introduced new variable partial_output to store the partial output
  dimensions after each horizontal split
  */
  Matrix partial_output;
  /*
  (final)output stores the final output dimensions after all the horizontal splits
  */
  Matrix output;
  int strides[2];
  int pads[4];
  int h_pads[4];
  // To take care of padding while doing horizontal splits
  // For first horizontal split bottom padding will always be 0
  // For the last horizontal split top padding will always be 0

  int kernels_original;
  int row_start;
  int row_end;

  // h_actual_splits and h_split_index - to take care of padding wrt different splits
  int h_actual_split;
  int h_split_index;
};

// this function reads all lines but takes into consideration only the required input
// if j==count , the program will not make str empty and return it once j>count
std::string read_filepath(std::ifstream& indata, int count) {
  std::string str;

  while (j <= count) {
    char num;
    while (indata) {
      std::getline(indata, str);

      if (j < count) {
        str = " ";
      }
      j++;
      break;
    }
  }
  return str;
}

int image_shares(Options* options, std::string p) {
  std::cout << "In image_shares : " << std::endl;
  std::ifstream temps;
  try {
    temps.open(p);
    if (temps) {
      std::cout << "File found\n";
    } else {
      std::cout << "Image shares file not found at " + p + "\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening the image share file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(temps)) {
      // file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Image share input file is empty.\n";
    return EXIT_FAILURE;
  }

  std::uint64_t chnls, rows, cols, val;
  try {
    chnls = read_file(temps);
    options->image_file.chnl = chnls;
    options->partial_output.chnl = chnls;
    rows = read_file(temps);
    // Sarthak/srishti/haritha changes
    // We are filling number of rows and columns corresponding to the partial image data

    options->partial_image_file.row = options->row_end - options->row_start + 1;
    options->image_file.row = rows;
    // std::cout << "r " << rows << " ";
    cols = read_file(temps);
    options->image_file.col = cols;
    options->partial_image_file.col = cols;
    options->partial_image_file.chnl = chnls;
    // std::cout << "c " << cols << "\n";
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while reading columns from image shares.\n";
    return EXIT_FAILURE;
  }

  std::cout << "chnls : " << chnls << " rows : " << options->image_file.row << "\n";
  for (int i = 0; i < chnls * options->image_file.row * cols; ++i) {
    try {
      uint64_t m1 = read_file(temps);
      options->image_file.Delta.push_back(m1);
      uint64_t m2 = read_file(temps);
      options->image_file.delta.push_back(m2);
    } catch (std::ifstream::failure e) {
      std::cerr << "Error while reading columns from image shares.\n";
      return EXIT_FAILURE;
    }
  }

  // Skipping the required number of rows from each channel and reading into partial_image
  // datastructure Sarthak/srishti/haritha changes
  for (int i = 0; i < chnls; ++i) {
    temps.seekg(std::ios::beg);

    int ch = read_file(temps);
    int r = read_file(temps);
    int c = read_file(temps);

    int t1 = (options->image_file.row * i + (options->row_start - 1)) * options->image_file.col;
    // std::cout << "Number of lines being skiped : " << t1 << "\n";

    for (int j = 0; j < t1; ++j) {
      temps.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for (int k = 0; k < (options->row_end - options->row_start + 1) * options->image_file.col;
         k++) {
      try {
        uint64_t m1 = read_file(temps);
        options->partial_image_file.Delta.push_back(m1);
        uint64_t m2 = read_file(temps);
        options->partial_image_file.delta.push_back(m2);
        // std::cout << "D : " << m1 << "d : "<< m2 << "--";
      } catch (std::ifstream::failure e) {
        std::cerr << "Error while reading columns from image shares.\n";
        return EXIT_FAILURE;
      }
    }
  }

  temps.close();
}
/* Depending on the rowstart and rowend index, the function will read the required number of
elements By skipping appropriate number of rows from each channel
*/

int W_shares(Options* options, std::string p) {
  std::ifstream indata;
  std::cout << "In ***************W_Shares****************" << std::endl;
  try {
    indata.open(p);
    if (std::ifstream(p)) {
      std::cout << "File found\n";
    } else {
      std::cout << "Weight shares file not found at " + p + "\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening the input weight share file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(indata)) {
      // file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Weight share input file is empty.\n";
    return EXIT_FAILURE;
  }

  // In W_shares file , the first line contains meta data about kernels in the following format
  // Kernels , channels , rows , columns , top_pad , left_pad , bottom_pad , right_pad , horizontal
  // stride , vertical stride
  int kernels, channels, rows, cols;
  try {
    kernels = options->kernel_end - options->kernel_start + 1;
    std::cout << "Kernel Start: " << options->kernel_start
              << "  Kernel End: " << options->kernel_end << std::endl;
    if (kernels < 1) {
      std::cerr << "\n Either row start or row end is not a feasible number.\n";
      return EXIT_FAILURE;
    }
    options->kernels_original = read_file(indata);
    if (options->kernel_end > options->kernels_original) {
      std::cerr << "\n Kernel end is not a feasible number.\n";
      return EXIT_FAILURE;
    }
    options->W_file.ker = kernels;
    channels = read_file(indata);
    options->W_file.chnl = channels;
    rows = read_file(indata);
    options->W_file.row = rows;
    cols = read_file(indata);
    options->W_file.col = cols;
    std::cout << "k " << kernels << " c " << channels << " r " << rows << " c " << cols << "\n";

    for (int i = 0; i < 4; i++) (options->pads)[i] = read_file(indata);

    std::cout << "pads 0 : " << options->pads[0] << "\n";
    std::cout << "pads 1 : " << options->pads[1] << "\n";
    std::cout << "pads 2 : " << options->pads[2] << "\n";
    std::cout << "pads 3 : " << options->pads[3] << "\n";

    // Sarthak changes
    // Padding information to be given conditionally depending on the horizontal split index
    // For eg For the first split, the padding information will be 1, 0, 1, 1
    // For the second split, the padding information will be 0, 0, 1, 1
    // For the last split, the padding information will be 0, 1, 1, 1
    // We need to take care here

    // Padding information : [T,L,B,R]
    if (options->h_split_index == 0) {
      (options->h_pads[0] = options->pads[0]);
    } else {
      options->h_pads[0] = 0;
    }
    if (options->h_split_index == options->h_actual_split - 1) {
      options->h_pads[2] = options->pads[2];
    } else {
      options->h_pads[2] = 0;
    }
    options->h_pads[1] = options->pads[1];
    options->h_pads[3] = options->pads[3];

    std::cout << "h pads : \n";
    std::cout << "TOP: " << options->h_pads[0] << " LEFT: " << options->h_pads[1]
              << " BOTTOM: " << options->h_pads[2] << " RIGHT: " << options->h_pads[3] << "\n";

    for (int i = 0; i < 2; i++) (options->strides)[i] = read_file(indata);
    std::cout << "strides " << (options->strides)[0] << " " << (options->strides)[1] << "\n";

    options->output.chnl = kernels;
    options->output.row = (options->image_file.row - options->W_file.row + options->pads[0] +
                           options->pads[2] + options->strides[0]) /
                          options->strides[0];
    options->output.col = (options->image_file.col - options->W_file.col + options->pads[1] +
                           options->pads[3] + options->strides[1]) /
                          options->strides[1];
    // Sarthak/srishti/haritha changes
    // Filling the partial output dims
    options->partial_output.chnl = kernels;
    int t;
    t = (options->row_end - options->row_start + 1);
    options->partial_output.row =
        (t - options->W_file.row + options->h_pads[0] + options->h_pads[2] + options->strides[0]) /
        options->strides[0];
    options->partial_output.col = (options->image_file.col - options->W_file.col +
                                   options->h_pads[1] + options->h_pads[3] + options->strides[1]) /
                                  options->strides[1];
    // << options->partial_output.row << "\n"
    // << options->partial_output.col << "\n";

  } catch (std::ifstream::failure e) {
    std::cerr << "Error while reading dimensions from weight shares.\n";
    return EXIT_FAILURE;
  }

  indata.seekg(std::ios::beg);
  for (int i = 0; i < ((options->kernel_start - 1) * channels * rows * cols) + 1;
       ++i) {  // Skip Dimensions and kernels
    indata.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  std::cout << kernels * channels * rows * cols << "\n";
  for (int i = 0; i < kernels * channels * rows * cols; ++i) {
    try {
      std::uint64_t m1 = read_file(indata);
      // std::cout << "m1 " << m1 << "\n";
      options->W_file.Delta.push_back(m1);

      std::uint64_t m2 = read_file(indata);
      // std::cout << "m2 " << m2 << "\n";
      options->W_file.delta.push_back(m2);

      // std::cout << "D : " << m1 << " d : " << m2 << "\n";
    } catch (std::ifstream::failure e) {
      std::cerr << "Error while reading input weight shares from file.\n";
      return EXIT_FAILURE;
    }
  }

  indata.close();
}

int B_shares(Options* options, std::string p) {
  std::cout << "In B_Shares" << std::endl;
  std::ifstream indata;
  try {
    if (std::ifstream(p)) {
      std::cout << "File found\n";
    } else {
      std::cout << "Bias shares file not found at " + p + "\n";
    }

    indata.open(p);
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening the input bias share file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(indata)) {
      // file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Bias share file is empty.\n";
    return EXIT_FAILURE;
  }

  int rows, cols;
  try {
    rows = read_file(indata);
    rows = options->kernel_end - options->kernel_start + 1;
    options->B_file.row = rows;
    std::cout << "r " << rows << " ";
    cols = read_file(indata);
    options->B_file.col = cols;
    std::cout << "c " << cols << "\n";
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while reading rows and columns from input bias share file.\n";
    return EXIT_FAILURE;
  }

  indata.seekg(std::ios::beg);
  for (int i = 0; i < ((options->kernel_start - 1) * cols) + 1; ++i) {
    indata.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  for (int i = 0; i < rows * cols; ++i) {
    try {
      uint64_t m1 = read_file(indata);
      uint64_t m2 = read_file(indata);
      for (int i = 0; i < options->partial_output.row * options->partial_output.col;
           i++)  // Blowing up Bias values by output row * output col no of times
      {
        options->B_file.Delta.push_back(m1);
        options->B_file.delta.push_back(m2);
      }
    } catch (std::ifstream::failure e) {
      std::cerr << "Error while reading the input bias share file.\n";
      return EXIT_FAILURE;
    }
  }
  indata.close();
  std::cout << "496\n B completed\n";
}

// changes made in this function
int file_read(Options* options) {
  std::string path = options->currentpath;
  std::string t1, i;
  t1 = path + "/server" + std::to_string(options->my_id) + "/" + options->imageprovider + "_" +
       std::to_string(options->my_id);

  image_shares(options, t1);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // model path --> builddebwithrelinfo_gcc/file_config_model0/1
  std::string t2 = path + "/" + options->modelpath;
  std::cout << t2 << "\n";
  std::ifstream file2;
  try {
    file2.open(t2);
    if (file2) {
      std::cout << "File found\n";
    } else {
      std::cout << "File not found at " + t2 + "\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening config_model file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(file2)) {
      // file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "file_config_model is empty.\n";
    return EXIT_FAILURE;
  }

  j = 0;
  //  skip 2(layer_id-1) and read first 2 lines
  int c = 2 * (options->layer_id - 1);
  std::string w = read_filepath(file2, c);
  std::cout << w << std::endl;
  // std::cout << "w:" << w << "\n";
  // std::cout << "j:" << j << "\n";
  std::string b = read_filepath(file2, c + 1);
  // std::cout << "b:" << b << "\n";
  file2.close();
  std::cout << "Reading W shares" << std::endl;
  W_shares(options, w);
  B_shares(options, b);
  std::cout << "Reading completed\n";
}

//////////////////////////////////////////////////////////////////////
std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("config-file-input", po::value<std::string>()->required(), "config file containing options")
    ("config-file-model", po::value<std::string>()->required(), "config file containing options")
    ("my-id", po::value<std::size_t>()->required(), "my party id")
    ("layer-id", po::value<std::size_t>()->required(), "layer id")
    ("party", po::value<std::vector<std::string>>()->multitoken(),
     "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("threads", po::value<std::size_t>()->default_value(0), "number of threads to use for gate evaluation")
    ("json", po::bool_switch()->default_value(false), "output data in JSON format")
    ("fractional-bits", po::value<std::size_t>()->default_value(16),
     "number of fractional bits for fixed-point arithmetic")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("current-path",po::value<std::string>()->required(), "current path build_debwithrelinfo")
    ("sync-between-setup-and-online", po::bool_switch()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false), "just build the circuit, but not execute it")
    ("kernel_start", po::value<int>()->required(), "Kernel start number for split convolution operation")
    ("kernel_end", po::value<int>()->required(), "Kernel end number for split convolution operation")
    ("row_start", po::value<int>()->required(), "Row start number for split convolution operation")
    ("row_end", po::value<int>()->required(), "Row end number for split convolution operation")
    ("h_actual_split", po::value<int>()->required(), "Actual split for convolution operation")
    ("h_split_index", po::value<int>()->required(), "Horizontal split index operation")
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
  options.currentpath = vm["current-path"].as<std::string>();
  //////////////////////////////////////////////////////////////////
  options.imageprovider = vm["config-file-input"].as<std::string>();
  options.modelpath = vm["config-file-model"].as<std::string>();
  options.layer_id = vm["layer-id"].as<std::size_t>();
  options.kernel_start = vm["kernel_start"].as<int>();
  options.kernel_end = vm["kernel_end"].as<int>();
  options.row_start = vm["row_start"].as<int>();
  options.row_end = vm["row_end"].as<int>();
  options.h_actual_split = vm["h_actual_split"].as<int>();
  options.h_split_index = vm["h_split_index"].as<int>();

  // SARTHAK TEMPORARY
  std::cout << "THIS IS ROW START AND ROW END : " << options.row_start << " " << options.row_end
            << std::endl;
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
    auto obj = MOTION::Statistics::to_json("tensor_gt_mul", run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("arithmetic_protocol", MOTION::ToString(options.arithmetic_protocol));
    obj.emplace("boolean_protocol", MOTION::ToString(options.boolean_protocol));
    obj.emplace("simd", options.num_simd);
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    // std::cout << obj << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats("tensor_gt_mul", run_time_stats, comm_stats);
  }
}

auto create_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  // std::cout << "Inside create_composite"
  //           << "\n";
  // retrieve the gate factories for the chosen protocols
  /*******************************************************************/

  auto& arithmetic_tof = backend.get_tensor_op_factory(options.arithmetic_protocol);
  auto& boolean_tof = backend.get_tensor_op_factory(MOTION::MPCProtocol::Yao);
  // Chainging the input dimensions to the partial image dimensions
  // read image file function
  // Sarthak changes
  const MOTION::tensor::TensorDimensions input_dims{
      .batch_size_ = 1,
      .num_channels_ = options.partial_image_file.chnl,
      .height_ = options.partial_image_file.row,
      .width_ = options.partial_image_file.col};
  const MOTION::tensor::TensorDimensions conv_weights_dims{.batch_size_ = options.W_file.ker,
                                                           .num_channels_ = options.W_file.chnl,
                                                           .height_ = options.W_file.row,
                                                           .width_ = options.W_file.col};
  // Bias Dimensions for convolution operation
  const MOTION::tensor::TensorDimensions CBias1_dims{options.partial_output.chnl, 1, 1, 1};
  // Bias Dimensions for addition operation which are obtained after blowing up the CBias1
  const MOTION::tensor::TensorDimensions CBias_dims{
      options.partial_output.chnl, options.partial_output.row, options.partial_output.col};

  const MOTION::tensor::Conv2DOp conv_op = {
      .kernel_shape_ = {options.W_file.ker, options.W_file.chnl, options.W_file.row,
                        options.W_file.col},
      .input_shape_ = {options.partial_image_file.chnl, options.partial_image_file.row,
                       options.partial_image_file.col},
      .output_shape_ = {options.partial_output.chnl, options.partial_output.row,
                        options.partial_output.col},
      .dilations_ = {1, 1},
      .pads_ = {options.h_pads[0], options.h_pads[1], options.h_pads[2], options.h_pads[3]},
      .strides_ = {options.strides[0], options.strides[1]}};

  std::cout << "----------------------------------" << std::endl
            << "Input Dimensions:" << std::endl;
  std::cout << "Batch Size: " << input_dims.batch_size_
            << "\nNumber of Channels: " << options.partial_image_file.chnl
            << "\nHeight of Matrix: " << options.partial_image_file.row
            << "\nWidth of Matrix: " << options.partial_image_file.col << std::endl;
  std::cout << "----------------------------------" << std::endl;
  std::cout << "Convolutional Weights:\n"
            << "Batch Size: " << conv_weights_dims.batch_size_
            << "\nNumber of Channels: " << options.W_file.chnl
            << "\nHeight of Matrix: " << options.W_file.row
            << "\nWidth of Matrix: " << options.W_file.col << std::endl;
  std::cout << "----------------------------------" << std::endl;
  std::cout << "Bias:\n"
            // << "Batch Size: " << conv_weights_dims.batch_size_
            // << "\nNumber of Channels: " << options.B_file.chnl
            << "\nHeight of B Matrix: " << options.B_file.row
            << "\nWidth of B Matrix: " << options.B_file.col << std::endl;
  std::cout << "----------------------------------" << std::endl;
  std::cout << "Output Dimensions:\n"
            << "Number of Channels: " << options.partial_output.chnl
            << "\nHeight of Matrix: " << options.partial_output.row
            << "\nWidth of Matrix: " << options.partial_output.col << std::endl;
  std::cout << "----------------------------------" << std::endl;

  MOTION::tensor::TensorCP tensor_X, tensor_CW1, tensor_CB1, tensor_CB, add_output;

  auto pairX = arithmetic_tof.make_arithmetic_64_tensor_input_shares(input_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_X =
      std::move(pairX.first);
  tensor_X = pairX.second;

  auto pairCW1 = arithmetic_tof.make_arithmetic_64_tensor_input_shares(conv_weights_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_CW1 =
      std::move(pairCW1.first);
  tensor_CW1 = pairCW1.second;

  auto pairCB1 = arithmetic_tof.make_arithmetic_64_tensor_input_shares(CBias1_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_CB1 =
      std::move(pairCB1.first);
  tensor_CB1 = pairCB1.second;

  auto pairCB = arithmetic_tof.make_arithmetic_64_tensor_input_shares(CBias_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_CB =
      std::move(pairCB.first);
  tensor_CB = pairCB.second;

  ///////////////////////////////////////////////////////////////
  input_promises_X[0].set_value(options.partial_image_file.Delta);
  input_promises_X[1].set_value(options.partial_image_file.delta);

  input_promises_CW1[0].set_value(options.W_file.Delta);
  input_promises_CW1[1].set_value(options.W_file.delta);

  // std::cout << "*******Weights Shares***********\n";
  // for (int i = 0; i < options.W_file.row * options.W_file.col; i++) {
  //   std::cout << options.W_file.Delta[i] << " " << options.W_file.delta[i] << "\n";
  // }

  // std::cout << "*******Partial Image Shares***********\n";
  // for (int i = 0; i < options.partial_image_file.row * options.partial_image_file.col; i++) {
  //   std::cout << options.partial_image_file.Delta[i] << " " <<
  //   options.partial_image_file.delta[i]
  //             << "\n";
  // }

  // std::cout << "*******Bias Shares***********\n";
  // for (int i = 0;
  //      i < options.partial_output.chnl * options.partial_output.row * options.partial_output.col;
  //      i++) {
  //   std::cout << options.B_file.Delta[i] << " " << options.B_file.delta[i] << "\n";
  // }

  input_promises_CB1[0].set_value(
      options.B_file.Delta);  // Dummy Bias values (5, 1, 1, 1) for mnist
  input_promises_CB1[1].set_value(options.B_file.delta);

  input_promises_CB[0].set_value(
      options.B_file.Delta);  // Blown up Bias values (845(5x13x13), 1) for mnist
  input_promises_CB[1].set_value(options.B_file.delta);

  //***********************************************************************

  // Convolution Operation
  auto conv_output = arithmetic_tof.make_tensor_conv2d_op(conv_op, tensor_X, tensor_CW1, tensor_CB1,
                                                          options.fractional_bits);
  // Bias Addition
  add_output = arithmetic_tof.make_tensor_add_op(conv_output, tensor_CB);

  ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> output_future;
  if (options.my_id == 0) {
    arithmetic_tof.make_arithmetic_tensor_output_other(add_output);
  } else {
    output_future = arithmetic_tof.make_arithmetic_64_tensor_output_my(add_output);
  }
  std::cout << "*******end of composite circuit*******\n";
  return output_future;
  /////////////////////////////////////////////////////////////////////////////////////
}

void genrt_cnn_outputshare(const Options& options) {
  std::string my_id = std::to_string(options.my_id);
  std::string o_path = options.currentpath + "/server" + my_id + "/cnn_outputshare_" + my_id;
  std::ofstream o_file;
  try {
    o_file.open(o_path);
    if (!o_file) {
      std::cout << "File not created at " + o_path + "\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening cnn_outputshare file.\n";
    return;
  }

  // Compute the convolution output dimensions and wrtie to o_file
  //
  o_file << options.kernels_original << " " <<  // Dimensions with channels
      options.output.row << " " << options.output.col << "\n";

  o_file.close();
  return;
}

void genrt_outputshare(const Options& options) {
  std::string my_id = std::to_string(options.my_id);
  std::string path = options.currentpath;
  std::string o_path = path + "/server" + my_id + "/final_outputshare_" + my_id;
  std::ofstream o_file;
  try {
    o_file.open(o_path);
    if (!o_file) {
      std::cout << "File not created at " + o_path + "\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening outputshare file.\n";
    return;
  }
  // Dimensions without channels
  o_file << options.kernels_original * options.output.row * options.output.col << " " << 1 << "\n";
  o_file.close();
  return;
}

void run_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto output_future = create_composite_circuit(options, backend);

  backend.run();
  if (options.my_id == 1) {
    auto main = output_future.get();

    //   std::vector<long double> mod_x;
    //   // std::string path = std::filesystem::current_path();
    //   std::string path = options.currentpath;
    //   string filename = path + "/" + "output_tensor";
    //   std::ofstream x;
    //   x.open(filename, std::ios_base::app);
    //   x << options.imageprovider << "\n";
    // int i = 1;
    std::cout << "output: \n";
    for (int k = 0; k < main.size(); ++k) {
      // std::cout << main[k] << " ";

      float temp =
          MOTION::new_fixed_point::decode<uint64_t, float>(main[k], options.fractional_bits);
      std::cout << temp << "\n";
    }
    std::cout << "******************************************" << std::endl;
  }
  // std::cout << "print output of the CNN " << main.size() << std::endl;
  // std::cout << "******************************************" << std::endl;
  // for (int i = 0; i < 5; ++i)
  // {
  //   std::cout << i << "******" << std::endl;
  //   for (int j = 0; j < 5; ++j)
  //   {
  //     for (int k = 0; k < 5; ++k)
  //       {
  //         long double temp =
  //       MOTION::new_fixed_point::decode<uint64_t, long double>(main[(i*5) + (j*13) + k ],
  //       options.fractional_bits);

  //       std::cṭout << temp << ",";

  return;
}

int main(int argc, char* argv[]) {
  // std::cout << "Inside main";
  auto options = parse_program_options(argc, argv);
  int WriteToFiles = 1;
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

    // Sarthak/srishti/haritha changes
    // We want to generate output shares file only once per layer
    if (options->kernel_start == 1 && options->row_start == 1) {
      genrt_cnn_outputshare(*options);
      genrt_outputshare(*options);
    }

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
          MOTION::Statistics::print_stats_short("tensor_gt_mul_test", run_time_stats, comm_stats);
      std::cout << "Execution time string:" << time_str << "\n";
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
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
