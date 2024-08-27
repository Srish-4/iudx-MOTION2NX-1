#include <bits/stdc++.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include "utility/new_fixed_point.h"

/*
./bin/reconstruct_cnn --current-path ${BASE_DIR}/build_debwithrelinfo_gcc
*/

using namespace boost::asio;
using ip::tcp;
using std::cout;
using std::endl;
using std::string;
namespace po = boost::program_options;
namespace fs = std::filesystem;
struct Shares {
  std::uint64_t Delta, delta;
};
struct Options {
  std::string currentpath;
  std::size_t fractional_bits;
};

std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
     ("current-path",po::value<std::string>()->required(), "current path build_debwithrelinfo")
    ("fractional-bits", po::value<std::size_t>()->default_value(13)) 
    ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  bool help = vm["help"].as<bool>();
  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  options.currentpath = vm["current-path"].as<std::string>();
  options.fractional_bits = vm["fractional-bits"].as<std::size_t>();

  return options;
}

int main(int argc, char* argv[]) {
  auto options = parse_program_options(argc, argv);
  std::string path = options->currentpath;
  try {
    // Reading contents from file
    std::string t1 = path + "/server0/outputshare_0";
    std::string t2 = path + "/server1/outputshare_1";

    std::cout << t1 << " " << t2 << "\n";

    std::ifstream file1, file2;
    float y;
    std::ofstream file;

    std::string t3 = path + "/cnn_result_layer1";
    file.open(t3);

    try {
      file1.open(t1);
      file2.open(t2);
      if (file1 && file2) {
        std::cout << " ";
      }
    } catch (std::ifstream::failure e) {
      std::cerr << "Error while opening the input files.\n";
      return EXIT_FAILURE;
    }
    std::uint64_t temp;
    // get input data
    // std::vector<float> data1

    int output_channels, output_rows, output_cols;
    // file1 >> output_channels >> output_rows >> output_cols;
    // file2 >> output_channels >> output_rows >> output_cols;
    file1 >> output_rows >> output_cols;
    file2 >> output_rows >> output_cols;

    // std::cout << output_channels << " " << output_rows << " " << output_cols << "\n";
    std::cout << output_rows << " " << output_cols << "\n";

    // std::cout << (output_channels * output_rows * output_cols) << "\n";

    // int output_size = (output_channels * output_rows * output_cols);
    int output_size = (output_rows * output_cols);

    Shares shares_data_0[output_size], shares_data_1[output_size];

    for (int i = 0; i < (output_rows * output_cols); i++) {
      file1 >> shares_data_0[i].Delta;
      file1 >> shares_data_0[i].delta;
      file2 >> shares_data_1[i].Delta;
      file2 >> shares_data_1[i].delta;

      // std::cout << shares_data_0[i].Delta << " " << shares_data_0[i].delta << " "
      //           << shares_data_1[i].Delta << " " << shares_data_1[i].delta << "\n";
      try {
        if (shares_data_0[i].Delta != shares_data_1[i].Delta)
          std::cout << "Error at " << i << " index \n";
      } catch (std::ifstream::failure e) {
        std::cerr << "Incorrect output shares.\n";
        return EXIT_FAILURE;
      }
      temp = shares_data_0[i].Delta - shares_data_0[i].delta - shares_data_1[i].delta;
      // y = temp / (pow(2, options->fractional_bits));
      y = MOTION::new_fixed_point::decode<uint64_t, float>(temp, 13);

      std::cout << y << "\n";
      file << y << "\n";
    }
    file.close();
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    std::cerr << "ERROR Caught !!"
              << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
