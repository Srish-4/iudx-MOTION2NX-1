#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <utility/new_fixed_point.h>
#include "utility/linear_algebra.h"

#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <random>

namespace po = boost::program_options;
using namespace std;

void convolution(vector<uint64_t> input, vector<uint64_t> weights, vector<uint64_t> bias,
                 int kernels, int channels, int rows, int cols, vector<int> pads,
                 vector<int> strides, int img_rows, int img_cols, bool dilate = false) {
  vector<vector<uint64_t>> kernel_segments;

  int temp = 0;
  for (int i = 0; i < kernels; i++) {
    auto first = weights.begin() + temp;
    temp += channels * rows * cols;
    auto last = weights.begin() + temp;
    vector<uint64_t> kernel(first, last);
    kernel_segments.push_back(kernel);
  }

  auto encoded_zero = MOTION::new_fixed_point::encode<uint64_t, float>(0, 13);

  vector<uint64_t> image;
  for (int c = 0; c < channels; c++) {
    for (int i = 0; i < pads[0]; i++) {
      for (int j = 0; j < img_cols + pads[1] + pads[3]; j++) image.push_back(encoded_zero);
    }

    for (int i = 0; i < img_rows; i++) {
      for (int j = 0; j < pads[1]; j++) image.push_back(encoded_zero);
      for (int j = 0; j < img_cols; j++)
        image.push_back(input[c * img_rows * img_cols + i * img_cols + j]);
      for (int j = 0; j < pads[3]; j++) image.push_back(encoded_zero);
    }

    for (int i = 0; i < pads[2]; i++) {
      for (int j = 0; j < img_cols + pads[1] + pads[3]; j++) image.push_back(encoded_zero);
    }
  }
  img_rows += pads[0] + pads[2];
  img_cols += pads[1] + pads[3];

  uint64_t output_chnls = kernels;
  uint64_t output_rows = (img_rows - rows + strides[0]) / strides[0];
  uint64_t output_columns = (img_cols - cols + strides[1]) / strides[1];

  vector<vector<uint64_t>> image_segments(output_rows * output_columns,
                                          vector<uint64_t>(channels * rows * cols));

  for (unsigned int i = 0; i < output_rows; i++) {
    for (unsigned int j = 0; j < output_columns; j++) {
      unsigned int row_start = i * strides[0];
      unsigned int col_start = j * strides[1];
      for (unsigned int k = 0; k < channels; k++) {
        for (unsigned int l = 0; l < rows * cols; l++) {
          unsigned int row = row_start + l / cols;
          unsigned int col = col_start + l % cols;
          image_segments[i * output_columns + j][k * rows * cols + l] =
              image[k * img_rows * img_cols + row * img_cols + col];
        }
      }
    }
  }
  vector<uint64_t> output(kernels * output_rows * output_columns);
  int j = 0;
  for (int k = 0; k < kernels; k++) {
    for (int i = 0; i < output_rows * output_columns; i++) {
      output[j] = (MOTION::matrix_multiply(1, kernel_segments[k].size(), 1, kernel_segments[k],
                                           image_segments[i]))[0];
      output[j] = MOTION::new_fixed_point::truncate(output[j], 13);
      output[j] += bias[k];
      output[j] = MOTION::new_fixed_point::decode<uint64_t, float>(output[j], 13);
      std::cout << output[j] << "\n";
      j++;
    }
  }
}

int main() {
  //   vector<uint64_t> input, vector<uint64_t> weights, int kernels, int channels, int rows, int
  //   cols,
  //       vector<int> pads, vector<int> strides, int img_rows, int img_cols

  // vector<uint64_t> input = {1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 2,
  //                           0, 0, 0, 2, 3, 0, 0, 0, 3, 0, 0, 0, 3};

  std::string home_dir = getenv("BASE_DIR");
  std::string path = home_dir + "/data/ImageProvider/images/X1.csv";
  std::ifstream file;

  file.open(path);
  std::vector<std::uint64_t> input;

  std::cout << "input: \n";
  while (file) {
    float temp;
    file >> temp;
    // std::cout << temp << "\n";
    auto temp_encoded = MOTION::new_fixed_point::encode<uint64_t, float>(temp, 13);
    input.push_back(temp_encoded);
    // input.push_back(temp);
  }
  std::cout << "\n";

  file.close();

  std::string path2 = home_dir + "/data/ModelProvider/W1.csv";
  std::cout << path2 << "\n";
  file.open(path2);
  std::vector<std::uint64_t> weights;

  std::cout << "weights: \n";
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      float temp = stold(cell);
      // std::cout << temp << "\n";
      auto temp_encoded = MOTION::new_fixed_point::encode<uint64_t, float>(temp, 13);

      weights.push_back(temp_encoded);
      // weights.push_back(temp);
    }
  }
  std::cout << "\n";

  file.close();

  std::string path3 = home_dir + "/data/ModelProvider/B1.csv";
  std::cout << path3 << "\n";
  file.open(path3);
  std::vector<std::uint64_t> bias;

  std::cout << "bias: \n";
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      float temp = stold(cell);
      std::cout << temp << "\n";
      auto temp_encoded = MOTION::new_fixed_point::encode<uint64_t, float>(temp, 13);

      bias.push_back(temp_encoded);
      // weights.push_back(temp);
    }
  }
  std::cout << "\n";

  file.close();

  // vector<uint64_t> input = {1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
  //                           2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2,
  //                           3, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3};

  // vector<uint64_t> weights = {1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5,
  //                             6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9};
  int kernels = 5;
  int channels = 3;
  int rows = 5;
  int cols = 5;
  vector<int> pads = {1, 1, 0, 0};
  vector<int> strides = {2, 2};
  int img_rows = 28;
  int img_cols = 28;

  convolution(input, weights, bias, kernels, channels, rows, cols, pads, strides, img_rows,
              img_cols);
}