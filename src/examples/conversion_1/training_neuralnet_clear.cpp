// Training neural network k
/*

./bin/training_neuralnet_clear --sample-size 20 --fractional-bits 13 --classes 10 --training-class 2
--sample-file sample_file --actual-label-file actual_label

*/
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <utility/new_fixed_point.h>
#include "utility/linear_algebra.h"

#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <random>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace po = boost::program_options;
using namespace std;

struct Options {
  std::size_t m;
  std::string actual_label_file;
  std::string sample_file;
  std::size_t frac_bits;
  std::size_t classes;
  std::size_t training_class;
};

// int g = 0;

void generate_random_numbers(std::vector<int>& sample_file, const Options& options) {
  std::random_device rd;   // Seed for the random number engine
  std::mt19937 gen(rd());  // Mersenne Twister random number generator
  static int r = 1;

  // r++;
  int minNumber = 1;
  int maxNumber = 5000;
  int numberOfRandomNumbers = options.m / options.classes;

  std::uniform_int_distribution<int> distribution(minNumber, maxNumber);

  // for (int i = 0; i < numberOfRandomNumbers; ++i) {
  //   // for (int i = 0; i < options.classes / options.m; ++i) {
  //   int randomNumber = distribution(gen);
  //   sample_file.push_back(r);
  //   // sample_file = r;
  // }

  for (int i = 0; i < numberOfRandomNumbers; ++i) {
    int randomNumber = distribution(gen);
    sample_file.push_back(randomNumber);
  }
}

class NeuralNetwork {
 private:
  std::vector<float> y_label;
  std::vector<float> x_transpose;
  std::vector<float> x;
  std::vector<std::uint64_t> encoded_y;
  std::vector<std::uint64_t> encoded_x_transpose;
  std::vector<std::uint64_t> encoded_x;
  std::vector<std::uint64_t> enc_w1;
  std::vector<std::uint64_t> enc_w2;
  std::vector<std::uint64_t> enc_b1;
  std::vector<std::uint64_t> enc_b2;
  std::vector<std::uint64_t> enc_z1;
  std::vector<std::uint64_t> enc_z2;
  std::vector<std::uint64_t> enc_a1;
  std::vector<std::uint64_t> enc_a2;
  std::vector<std::uint64_t> dw1;
  std::vector<std::uint64_t> dw2;
  std::vector<std::uint64_t> db1;
  std::vector<std::uint64_t> db2;
  std::vector<std::uint64_t> rate_dw1;
  std::vector<std::uint64_t> rate_dw2;
  std::vector<std::uint64_t> rate_db1;
  std::vector<std::uint64_t> rate_db2;

  friend void activation_function(std::vector<std::uint64_t>& input, std::string function_type,
                                  Options& options, const NeuralNetwork& network);

 public:
  NeuralNetwork(int n0, int n1, int n2, Options& options) {
    //********************************************************************************
    // std::random_device rd;   // Seed for the random number engine
    // std::mt19937 gen(rd());  // Mersenne Twister random number generator
    // int minNumber = 0;
    // int maxNumber = 8192;

    // std::uniform_int_distribution<std::uint64_t> distribution(minNumber, maxNumber);

    std::ifstream file;
    std::string home_dir = getenv("BASE_DIR");

    std::string W1_csv = home_dir + "/data/ModelProvider/W1_enc.csv";
    std::string B1_csv = home_dir + "/data/ModelProvider/B1_enc.csv";
    std::string B2_csv = home_dir + "/data/ModelProvider/B2_enc.csv";
    std::string W2_csv = home_dir + "/data/ModelProvider/W2_enc.csv";

    file.open(W1_csv);
    std::string str;
    while (std::getline(file, str)) {
      std::stringstream obj(str);
      std::string temp;
      while (std::getline(obj, temp, ',')) {
        auto input = std::stof(temp);
        enc_w1.push_back(input);
      }
    }
    file.close();

    file.open(B1_csv);
    while (std::getline(file, str)) {
      std::stringstream obj(str);
      std::string temp;
      while (std::getline(obj, temp, ',')) {
        auto input = std::stof(temp);
        enc_b1.push_back(input);
      }
    }
    file.close();

    file.open(W2_csv);
    while (std::getline(file, str)) {
      std::stringstream obj(str);
      std::string temp;
      while (std::getline(obj, temp, ',')) {
        auto input = std::stof(temp);
        enc_w2.push_back(input);
      }
    }
    file.close();

    file.open(B2_csv);
    while (std::getline(file, str)) {
      std::stringstream obj(str);
      std::string temp;
      while (std::getline(obj, temp, ',')) {
        auto input = std::stof(temp);
        enc_b2.push_back(input);
      }
    }
    file.close();

    //****************************************************************************************

    // std::cout << "W1 size : " << enc_w1.size() << "\n";
    // std::cout << "B1 size : " << enc_b1.size() << "\n";
    // std::cout << "W2 size : " << enc_w2.size() << "\n";
    // std::cout << "B2 size : " << enc_b2.size() << "\n";

    // enc_w1.assign(n1 * n0, 0);  // 2*2
    // float arr[8] = {-1, -1, 1, 2, 3, 4, 5, 6};
    // std::cout << "enc_w1:\n";
    // for (int i = 0; i < enc_w1.size(); i++) {
    //   auto t = MOTION::new_fixed_point::encode<std::uint64_t, float>(arr[i], 13);
    //   std::cout << t << " ";
    //   enc_w1[i] = t;
    // }
    // std::cout << "\n";

    // enc_w2.assign(n2 * n1, 0);  // 3*2
    // float arr2[12] = {1, 2, 3, -5, 2, 3, 4, 5, 3, 4, 5, 6};
    // std::cout << "enc_w2:\n";
    // for (int i = 0; i < enc_w2.size(); i++) {
    //   auto t = MOTION::new_fixed_point::encode<std::uint64_t, float>(arr2[i], 13);
    //   std::cout << t << " ";
    //   enc_w2[i] = t;
    // }
    // std::cout << "\n";

    // enc_b1.assign(n1 * 1, 0);  // 256 * 1
    // // temporary 2*1
    // float arr3[4] = {0.1, 0.2, 0.3, 0.4};
    // std::cout << "enc_b1:\n";
    // for (int i = 0; i < enc_b1.size(); i++) {
    //   auto t = MOTION::new_fixed_point::encode<std::uint64_t, float>(arr3[i], 13);
    //   std::cout << t << " ";
    //   enc_b1[i] = t;
    // }

    // enc_b2.assign(n2 * 1, 0);  // 10 * 1
    // // temporary  3*1
    // float arr4[3] = {0.3, 0.6, 0.9};
    // std::cout << "enc_b2:\n";
    // for (int i = 0; i < enc_b2.size(); i++) {
    //   auto t = MOTION::new_fixed_point::encode<std::uint64_t, float>(arr4[i], 13);
    //   std::cout << t << " ";
    //   enc_b2[i] = t;
    // }
    // std::cout << "\n\n";

    y_label.assign(n2 * options.m, 0);  // 10 * m

    // 3*3
    // y_label[0] = 1;
    // y_label[1] = 0;
    // y_label[2] = 0;
    // y_label[3] = 0;
    // y_label[4] = 1;
    // y_label[5] = 0;
    // y_label[6] = 0;
    // y_label[7] = 0;
    // y_label[8] = 1;
  }

  void read_input(Options& options) {
    // std::cout << "x size before reading : ";
    // std::cout << x.size() << " " << x_transpose.size() << "\n";
    std::string home_dir = getenv("BASE_DIR");

    std::vector<int> sample_file;
    for (int i = 0; i < options.classes; i++) {
      for (int j = 0; j < (options.m / options.classes); j++) {
        sample_file.push_back(j + 1);
      }
    }

    // for testing purpose reading files in a manner
    // int count = 0;
    // int k = g;
    // for (int i = 0; i < options.classes; i++) {
    //   while (count < (options.m / options.classes)) {
    //     k = k + 1;
    //     sample_file.push_back(k);
    //     count++;
    //   }
    //   if (i == options.classes - 1) {
    //     g = k;
    //     break;
    //   }
    //   k = g;
    //   count = 0;
    // }

    // using randomly choosen training data
    //  for (int i = 0; i < options.classes; i++) {
    //    generate_random_numbers(sample_file, options);
    //  }

    std::ifstream file;
    std::string path = home_dir + "/data/ImageProvider/sample_data/";
    std::string input_path;

    for (int i = 0; i < options.m; i++) {
      input_path = path + "images_folder" +
                   std::to_string((int)(i / (options.m / options.classes))) + "/X" +
                   std::to_string(sample_file[i]) + ".csv";

      std::cout << input_path << "\n";
      int class_no = (int)(i / (options.m / options.classes));
      int temp = class_no + i * options.classes;
      // std::cout << "y label: " << temp << "\n";
      y_label[temp] = 1;

      // std::cout << "x_transpose : \n";
      // // x_transpose is m * 784
      file.open(input_path);
      std::string str;
      while (std::getline(file, str)) {
        std::stringstream obj(str);
        std::string temp;
        while (std::getline(obj, temp, ',')) {
          auto input = std::stof(temp);
          // std::cout << input << " ";
          x_transpose.push_back(input);
        }
      }
      // std::cout << "\n";
      file.close();
    }

    // x_transpose.assign(6, 0);
    // std::cout << "x transpose : ";
    // float arr2[6] = {0, 1, 1, 0, 1, 1};
    // for (int i = 0; i < x_transpose.size(); i++) {
    //   x_transpose[i] = arr2[i];
    //   std::cout << x_transpose[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "x transpose size : " << x_transpose.size() << "\n";
    // std::cout << "y label size : " << y_label.size() << "\n";

    // std::cout << "y_label: \n";
    // for (int i = 0; i < y_label.size(); i++) {
    //   std::cout << y_label[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "y label display :  \n";
    // for (int i = 0; i < options.classes; ++i) {
    //   for (int j = 0; j < options.m; j++) {
    //     auto data = y_label[j * options.classes + i];
    //     std::cout << data << " ";
    //   }
    //   std::cout << "\n";
    // }

    int size_single_input = x_transpose.size() / options.m;
    // std::cout << "x : \n";
    //  // x is 784*m
    for (int i = 0; i < size_single_input; ++i) {
      for (int j = 0; j < options.m; j++) {
        auto data = x_transpose[j * size_single_input + i];
        // std::cout << data << " ";
        x.push_back(data);
      }
      // std::cout << "\n";
    }
    // std::cout << "\n\n";
    std::cout << "x size after reading : ";
    std::cout << x.size() << " " << x_transpose.size() << "\n";
  }

  void encoded_inputs(Options& options) {
    int frac_bits = options.frac_bits;

    encoded_y.assign(y_label.size(), 0);
    encoded_x_transpose.assign(x_transpose.size(), 0);
    encoded_x.assign(x.size(), 0);

    std::transform(y_label.begin(), y_label.end(), encoded_y.begin(), [frac_bits](auto j) {
      // return MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
      auto temp = MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
      // std::cout << temp << " ";
      return temp;
    });
    // std::cout << "\n";

    std::transform(x.begin(), x.end(), encoded_x.begin(), [frac_bits](auto j) {
      // return MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
      auto temp = MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
      // std::cout << temp << " ";
      return temp;
    });
    // std::cout << "\n";

    std::transform(x_transpose.begin(), x_transpose.end(), encoded_x_transpose.begin(),
                   [frac_bits](auto j) {
                     return MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
                   });
  }

  void forward_propogation(Options& options, int n1, int n2, bool test) {
    std::cout << "************forward propogation**************\n";
    int frac_bits = options.frac_bits;
    int m = options.m;
    std::vector<float> test_input;
    std::vector<int> test_label;

    // std::cout << "w1 size : " << enc_w1.size() << "\n";
    // std::cout << "b1 size : " << enc_b1.size() << "\n";
    // std::cout << "w2 size : " << enc_w2.size() << "\n";
    // std::cout << "b2 size : " << enc_b2.size() << "\n";
    // std::cout << "x size : " << x.size() << "\n";

    // std::cout << "B2 in forward propogation :- \n";
    // for (int i = 0; i < enc_b2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "B1 in forward propogation :- \n";
    // for (int i = 0; i < enc_b1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "m:  " << m << "\n";

    int n0 = encoded_x.size() / m;
    enc_z1.assign(n1 * m, 0);

    // std::cout << "encoded x:\n";

    // for (int i = 0; i < encoded_x.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(encoded_x[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    enc_z1 = MOTION::matrix_multiply(n1, n0, m, enc_w1, encoded_x);
    std::transform(enc_z1.begin(), enc_z1.end(), enc_z1.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "z1 after multiplying w1 and x \n";
    // for (int i = 0; i < enc_z1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << "z1_encoded size : " << enc_z1.size() << "\n";
    // std::cout << "b1 size : " << enc_b1.size() << "\n";

    for (int i = 0; i < options.m; i++) {
      for (int j = 0; j < n1; j++) {
        enc_z1[i + m * j] = enc_z1[i + m * j] + enc_b1[j];
      }
    }

    // std::cout << "z1 after adding b1 \n";
    // for (int i = 0; i < enc_z1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << "z1_encoded size : " << enc_z1.size() << "\n";

    enc_a1.assign(enc_z1.size(), 0);

    enc_a1 = enc_z1;

    // activation_function(enc_a1, "relu", options);
    activation_function(enc_a1, "relu", options, *this);

    // std::cout << "a1 size : " << enc_a1.size() << "\n";

    // std::cout << "a1 after activation function: \n";
    // for (int i = 0; i < enc_a1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_a1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << m << "\n";
    // std::cout << enc_w2.size() << "\n";

    enc_z2.assign(n2 * m, 0);

    std::cout << n2 << "<--n2 \n";

    enc_z2 = MOTION::matrix_multiply(n2, n1, m, enc_w2, enc_a1);

    std::transform(enc_z2.begin(), enc_z2.end(), enc_z2.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "z2_encoded size : " << enc_z2.size() << "\n";

    std::cout << "z2 after multiplying w2 and a1 \n";
    for (int i = 0; i < enc_z2.size(); i++) {
      std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z2[i], frac_bits)
                << " ";
    }
    std::cout << "\n\n";

    // std::cout << "b2 size : " << enc_b2.size() << "\n";

    for (int i = 0; i < options.m; i++) {
      for (int j = 0; j < n2; j++) {
        enc_z2[i + m * j] = enc_z2[i + m * j] + enc_b2[j];
      }
    }

    // std::cout << "z2 after adding b2 \n";
    // for (int i = 0; i < enc_z2.size(); i++) {
    //   if (i % m == 0) {
    //     std::cout << "\n";
    //   }
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    enc_a2.assign(enc_z2.size(), 0);
    enc_a2 = enc_z2;

    activation_function(enc_a2, "sigmoid", options, *this);

    // std::cout << "a2 size : " << enc_a2.size() << "\n";

    // std::cout << "a2 after activation function: \n";
    // for (int i = 0; i < enc_a2.size(); i++) {
    //   if (i % m == 0) cout << "\n";
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_a2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    std::cout << "a2 transpose after activation function: \n";

    int size_single_a2 = enc_a2.size() / m;
    // std::cout << " size single a1 : " << size_single_a1 << "\n";
    std::vector<std::uint64_t> transpose_a2;
    transpose_a2.assign(enc_a2.size(), 0);

    int k;
    int count;
    int index = 0;
    for (int j = 0; j < m; j++) {
      count = 0;
      k = j;

      while (count != size_single_a2) {
        auto temp = enc_a2[k];

        std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(temp, frac_bits) << " ";
        transpose_a2[index++] = temp;
        k = k + m;
        count++;
      }
      std::cout << "\n";
    }
  }

  void cost_function(Options& options, int n1, int n2, int n0) {
    std::cout << "\n*************Cost Function**********\n";

    int m = options.m;
    int frac_bits = options.frac_bits;

    // y is pushed like this 10000000000 , 1000000000 , 0100000000 ..
    // y transpose is read like this 110000..0 , 00110000..0

    // std::cout << "encoded y transpose :  ";
    std::vector<std::uint64_t> encoded_y_transpose;
    for (int i = 0; i < options.classes; ++i) {
      for (int j = 0; j < m; j++) {
        auto data = encoded_y[j * options.classes + i];
        // std::cout << data << " ";
        encoded_y_transpose.push_back(data);
      }
      // std::cout << "\n";
    }

    std::vector<std::uint64_t> dz2(enc_a2.size(), 0);
    std::transform(enc_a2.begin(), enc_a2.end(), encoded_y_transpose.begin(), dz2.begin(),
                   std::minus{});

    // std::cout << "dz2 after a2-y :  ";
    // for (int i = 0; i < dz2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(dz2[i], frac_bits) << "
    //   ";
    // }
    // std::cout << "\n";

    // //
    // #############################################################################################################

    std::uint64_t encoded_m = MOTION::new_fixed_point::encode<std::uint64_t, float>(m, frac_bits);

    db2.assign(n2 * 1, 0);

    // std::cout << "db2:  ";
    for (int i = 0; i < options.classes; i++) {
      uint64_t sum = 0;
      for (int j = 0; j < m; j++) {
        auto temp = dz2[j + m * i];
        // std::cout << temp << " ";
        sum = sum + temp;
      }
      auto temp1 = MOTION::new_fixed_point::decode<std::uint64_t, float>(sum, frac_bits);
      // std::cout << "sum:" << temp1 << " ";

      auto temp2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(temp1 / m, frac_bits);

      db2[i] = temp2;

      // std::cout << "db2:" << MOTION::new_fixed_point::decode<std::uint64_t, float>(temp2,
      // frac_bits)
      //           << " ";
    }
    // // std::cout << "\n\n";
    // std::cout << "db2 size:" << db2.size() << "\n";
    // std::cout << "db2:  ";
    // for (int i = 0; i < db2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(db2[i], frac_bits) << "
    //   ";
    // }
    // std::cout << "\n";

    // //
    // ############################################################################################################

    int size_single_a1 = enc_a1.size() / m;
    // std::cout << " size single a1 : " << size_single_a1 << "\n";
    std::vector<std::uint64_t> transpose_a1;
    transpose_a1.assign(enc_a1.size(), 0);

    // // a1 is 256*m
    // // a1 transpose is m*256
    // std::cout << "a1 transpose \n";
    int k;
    int count;
    int index = 0;
    for (int j = 0; j < m; j++) {
      count = 0;
      k = j;

      while (count != size_single_a1) {
        auto temp = enc_a1[k];

        // std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(temp, frac_bits) << "
        // ";
        transpose_a1[index++] = temp;
        k = k + m;
        count++;
      }
      // std::cout << "\n";
    }

    // std::cout << "size of transpose a1 :" << transpose_a1.size() << "\n";

    dw2 = MOTION::matrix_multiply(n2, m, n1, dz2, transpose_a1);
    std::transform(dw2.begin(), dw2.end(), dw2.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "dw2 size: " << dw2.size() << "\n";

    int sum = 0;
    // std::cout << "\n\ndw2 after multiplying dz2 and transpose a1 :  ";
    for (int i = 0; i < dw2.size(); i++) {
      auto temp = MOTION::new_fixed_point::decode<std::uint64_t, float>(dw2[i], frac_bits);
      dw2[i] = MOTION::new_fixed_point::encode<std::uint64_t, float>(temp / m, frac_bits);
      // std::cout << temp / m << " ";
      int ab = abs(temp / m);
      sum += ab;
    }
    std::cout << "absolute sum of dw2:" << sum << "\n";

    // // end of db2 and dw2
    // ##############################################################################################################

    // // enc_w2 is 10*256
    int size_single_input = enc_w2.size() / n2;
    std::vector<std::uint64_t> transpose_enc_w2;
    transpose_enc_w2.assign(enc_w2.size(), 0);

    // std::cout << "transpose encoded w2 is :\n";
    index = 0;
    // // transpose_enc_w2 is 256*10
    for (int i = 0; i < size_single_input; ++i) {
      for (int j = 0; j < n2; j++) {
        auto data = enc_w2[j * size_single_input + i];
        // std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(data, frac_bits) << "
        // ";
        transpose_enc_w2[index++] = data;
      }
      // std::cout << "\n";
    }

    // std::cout << "transposed w2 size:" << transpose_enc_w2.size() << "\n";

    std::vector<std::uint64_t> temp_vect(n1 * m, 0);
    temp_vect = MOTION::matrix_multiply(n1, n2, m, transpose_enc_w2, dz2);
    std::transform(temp_vect.begin(), temp_vect.end(), temp_vect.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "\n\ntemp after multiplying w2 transpose and dz2 :  ";
    // for (int i = 0; i < temp_vect.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(temp_vect[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    auto one_encoded = MOTION::new_fixed_point::encode<std::uint64_t, float>(1, frac_bits);

    std::vector<std::uint64_t> D_a1;
    D_a1.assign(enc_a1.size(), 0);
    index = 0;
    for (int i = 0; i < enc_a1.size(); i++) {
      D_a1[index++] = (enc_a1[i] > 0) ? one_encoded : 0;
    }

    // std::cout << "\n\nD_z1 after differentiation  :  ";
    // for (int i = 0; i < D_a1.size(); i++) {
    //   std::cout << D_a1[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "D_z1 size : " << D_a1.size() << "\n";

    std::transform(temp_vect.begin(), temp_vect.end(), D_a1.begin(), temp_vect.begin(),
                   std::multiplies{});
    std::transform(temp_vect.begin(), temp_vect.end(), temp_vect.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // //##########################################################################################################################

    int single_size_temp_vect = temp_vect.size() / m;

    db1.assign(n1 * 1, 0);

    // std::cout << "db1: ";
    for (int i = 0; i < single_size_temp_vect; i++) {
      uint64_t sum = 0;
      for (int j = 0; j < m; j++) {
        auto temp = temp_vect[j + m * i];
        // std::cout << temp << " ";
        sum = sum + temp;
      }
      auto temp1 = MOTION::new_fixed_point::decode<std::uint64_t, float>(sum, frac_bits);
      // std::cout << "sum:" << temp1 << " ";

      auto temp2 = MOTION::new_fixed_point::encode<std::uint64_t, float>(temp1 / m, frac_bits);

      db1[i] = temp2;

      // std::cout << "db1:" << MOTION::new_fixed_point::decode<std::uint64_t, float>(temp2,
      // frac_bits)
      //           << " ";
    }

    // std::cout << "\n\n";

    // std::cout << "db1 size : " << db1.size() << "\n";
    // std::cout << "db1: ";
    // for (int i = 0; i < db1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(db1[i], frac_bits) << "
    //   ";
    // }
    // std::cout << "\n";

    // //######################################################################################################################
    dw1.assign(n1 * n0, 0);
    dw1 = MOTION::matrix_multiply(n1, m, n0, temp_vect, encoded_x_transpose);
    std::transform(dw1.begin(), dw1.end(), dw1.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "\n\ndw1 after multiplying temp_vect and transpose X :";
    for (int i = 0; i < dw1.size(); i++) {
      auto temp = MOTION::new_fixed_point::decode<std::uint64_t, float>(dw1[i], frac_bits);
      dw1[i] = MOTION::new_fixed_point::encode<std::uint64_t, float>(temp / m, frac_bits);
      // std::cout << temp / m << " ";
    }
    // std::cout << "\n";
  }

  void backward_propogation(Options& options, float alpha) {
    std::cout << "\n\n***************backward propogation***************\n\n";
    int frac_bits = options.frac_bits;

    auto encoded_alpha =
        MOTION::new_fixed_point::encode<std::uint64_t, float>(alpha, options.frac_bits);

    std::vector<std::uint64_t> rate_dw1, rate_dw2, rate_db1;
    std::vector<std::uint64_t> rate_db2;

    // std::cout << dw1.size() << " " << dw2.size() << " " << db1.size() << " " << db2.size() <<
    // "\n";

    rate_dw1.assign(dw1.size(), 0);
    rate_dw2.assign(dw2.size(), 0);
    rate_db1.assign(db1.size(), 0);
    rate_db2.assign(db2.size(), 0);

    std::transform(dw1.begin(), dw1.end(), rate_dw1.begin(),
                   [encoded_alpha](auto j) { return encoded_alpha * j; });

    std::transform(dw2.begin(), dw2.end(), rate_dw2.begin(),
                   [encoded_alpha](auto j) { return encoded_alpha * j; });

    std::transform(db1.begin(), db1.end(), rate_db1.begin(),
                   [encoded_alpha](auto j) { return encoded_alpha * j; });

    std::transform(db2.begin(), db2.end(), rate_db2.begin(),
                   [encoded_alpha](auto j) { return encoded_alpha * j; });

    std::transform(rate_dw1.begin(), rate_dw1.end(), rate_dw1.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "\n\nrate_dw1 after multiplying dw1 and alpha  : \n";
    // std::cout << "rate_dw1 size : " << rate_dw1.size() << "\n";
    // for (int i = 0; i < rate_dw1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(rate_dw1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    std::transform(rate_dw2.begin(), rate_dw2.end(), rate_dw2.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "\n\nrate_dw2 after multiplying dw2 and alpha  : \n";
    // std::cout << "rate_dw2 size : " << rate_dw2.size() << "\n";
    // for (int i = 0; i < rate_dw2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(rate_dw2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    std::transform(rate_db2.begin(), rate_db2.end(), rate_db2.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "\n\nrate_db2 after multiplying db2 and alpha  : \n";
    // std::cout << "rate_db2 size : " << rate_db2.size() << "\n";
    // for (int i = 0; i < rate_db2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(rate_db2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    std::transform(rate_db1.begin(), rate_db1.end(), rate_db1.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "\n\nrate_db1 after multiplying db1 and alpha  : \n";
    // std::cout << "rate_db1 size : " << rate_db1.size() << "\n";
    // for (int i = 0; i < rate_db1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(rate_db1[i],
    //   frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    std::transform(enc_w1.begin(), enc_w1.end(), rate_dw1.begin(), enc_w1.begin(), std::minus{});

    std::cout << "\n\nfinal enc_w1  : \n";
    for (int i = 0; i < enc_w1.size(); i++) {
      std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_w1[i], frac_bits)
                << ",";
    }
    std::cout << "\n";

    std::transform(enc_b1.begin(), enc_b1.end(), rate_db1.begin(), enc_b1.begin(), std::minus{});

    std::cout << "\n\nfinal enc_b1  : \n";
    for (int i = 0; i < enc_b1.size(); i++) {
      std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b1[i], frac_bits)
                << ",";
    }
    std::cout << "\n";

    std::transform(enc_w2.begin(), enc_w2.end(), rate_dw2.begin(), enc_w2.begin(), std::minus{});

    std::cout << "\n\nfinal enc_w2  : \n";
    for (int i = 0; i < enc_w2.size(); i++) {
      std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_w2[i], frac_bits)
                << ",";
    }
    std::cout << "\n";

    std::transform(enc_b2.begin(), enc_b2.end(), rate_db2.begin(), enc_b2.begin(), std::minus{});

    std::cout << "\n\nfinal enc_b2  : \n";
    for (int i = 0; i < enc_b2.size(); i++) {
      std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b2[i], frac_bits)
                << ",";
    }
    std::cout << "\n";

    x.clear();
    x_transpose.clear();
  }

  void print_weights_biases(Options& options) {
    int frac_bits = options.frac_bits;
    std::cout << "\n***************Printing final weights and biases**************\n";

    std::ofstream file;

    std::string home_dir = getenv("BASE_DIR");

    std::string path = home_dir + "/build_debwithrelinfo_gcc";

    file.open(path + "/W1_enc", std::ios_base::out);

    std::cout << "\n\n***************  final enc_w1  :**************** \n";
    std::cout << "w1 size :" << enc_w1.size() << "\n";
    for (int i = 0; i < enc_w1.size(); i++) {
      file << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_w1[i], frac_bits) << "\n";
    }
    std::cout << "\n";
    file.close();

    file.open(path + "/B1_enc", std::ios_base::out);
    std::cout << "\n\n*******************  final enc_b1  :*********************** \n";
    for (int i = 0; i < enc_b1.size(); i++) {
      file << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b1[i], frac_bits) << "\n";
      // file << enc_b1[i] << "\n";
    }
    std::cout << "\n";
    file.close();

    file.open(path + "/W2_enc", std::ios_base::out);
    std::cout << "\n\n******************** final enc_w2  :************************* \n";
    for (int i = 0; i < enc_w2.size(); i++) {
      file << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_w2[i], frac_bits) << "\n";
    }
    std::cout << "\n";
    file.close();

    file.open(path + "/B2_enc", std::ios_base::out);
    std::cout << "\n\n**************** final enc_b2  :************************** \n";
    for (int i = 0; i < enc_b2.size(); i++) {
      file << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b2[i], frac_bits) << "\n";
    }
    std::cout << "\n";
    file.close();
  }

  std::uint64_t read_file(std::ifstream& pro) {
    std::string str;
    char num;
    while (pro >> std::noskipws >> num) {
      if (num != '\n') {
        str.push_back(num);
      } else {
        break;
      }
    }

    std::string::size_type sz = 0;
    std::uint64_t ret = (uint64_t)std::stoull(str, &sz, 0);
    return ret;
  }

  void read_test_inputs(std::vector<float>& test_input, std::vector<int>& test_label, int test_size,
                        Options& options) {
    std::vector<float> test_transpose_input;

    std::string home_dir = getenv("BASE_DIR");
    std::ifstream file;

    std::string path = home_dir + "/data/ImageProvider/images_actualanswer";
    std::string input_path;

    for (int i = 1; i <= test_size; i++) {
      input_path = path + "/X" + std::to_string(i) + ".csv";
      // std::cout << input_path << "\n";
      file.open(input_path);
      std::string str;
      int line = 0;
      while (std::getline(file, str)) {
        std::stringstream obj(str);
        std::string temp;
        while (std::getline(obj, temp, ',')) {
          if (line == 0) {
            auto input = std::stof(temp);
            test_label.push_back(input);
          } else {
            auto input = std::stof(temp);
            test_transpose_input.push_back(input);
          }
        }
        line++;
      }
      file.close();
    }

    std::cout << "test input transpose size : " << test_transpose_input.size() << "\n";
    // std::cout << "test_label size :" << test_label.size() << "\n";

    int size_single_input = test_transpose_input.size() / test_size;

    // std::cout << "test_input : \n";
    // // x is 784*m
    for (int i = 0; i < size_single_input; ++i) {
      for (int j = 0; j < test_size; j++) {
        auto data = test_transpose_input[j * size_single_input + i];
        // std::cout << data << " ";
        test_input.push_back(data);
      }
    }

    // // TESTING WITH TRAINING DATA SET

    // std::vector<int> sample_file;
    // for (int i = 0; i < test_size; i++) {
    //   generate_random_numbers(sample_file, options);
    // }

    // std::string path = home_dir + "/data/ImageProvider/sample_data/";
    // std::string input_path;

    // for (int i = 0; i < test_size; i++) {
    //   input_path = path + "images_folder" +
    //                std::to_string((int)(i / (test_size / options.classes))) + "/X" +
    //                std::to_string(sample_file[i]) + ".csv";

    //   auto k = i / (test_size / options.classes);
    //   test_label.push_back(k);

    //   // std::cout << input_path << "\n";

    //   file.open(input_path);
    //   std::string str;
    //   while (std::getline(file, str)) {
    //     std::stringstream obj(str);
    //     std::string temp;
    //     while (std::getline(obj, temp, ',')) {
    //       auto input = std::stof(temp);
    //       // std::cout << input << " ";
    //       test_transpose_input.push_back(input);
    //     }
    //   }
    //   // std::cout << "\n";
    //   file.close();
    // }

    // std::cout << "test input transpose size : " << test_transpose_input.size() << "\n";

    // int size_single_input = test_transpose_input.size() / test_size;
    // std::cout << "x : \n";
    // //  // x is 784*m
    // for (int i = 0; i < size_single_input; ++i) {
    //   for (int j = 0; j < test_size; j++) {
    //     auto data = test_transpose_input[j * size_single_input + i];
    //     // std::cout << data << " ";
    //     test_input.push_back(data);
    //   }
    //   // std::cout << "\n";
    // }
  }

  void test_accuracy(Options& options, int n1, int n2) {
    std::cout << "\n*************Testing Accuracy*************\n";
    int frac_bits = options.frac_bits;
    std::vector<float> test_input;
    std::vector<int> test_label;

    int m = 10000;
    read_test_inputs(test_input, test_label, m, options);

    // std::cout << "w1 size : " << enc_w1.size() << "\n";
    // std::cout << "b1 size : " << enc_b1.size() << "\n";
    // std::cout << "w2 size : " << enc_w2.size() << "\n";
    // std::cout << "b2 size : " << enc_b2.size() << "\n";

    // std::cout << "\n\nenc_b1 in test accuracy  : \n";
    // for (int i = 0; i < enc_b1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "\n\nenc_b2 in test accuracy  : \n";
    // for (int i = 0; i < enc_b2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_b2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "\n\nenc_w1 in test accuracy  : \n";
    // for (int i = 0; i < enc_w1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_w1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "\n\nenc_w2 in test accuracy  : \n";
    // for (int i = 0; i < enc_w2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_w2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "test_input size : " << test_input.size() << "\n";
    // std::cout << "test_label size :" << test_label.size() << "\n";

    std::vector<std::uint64_t> encoded_test_input(test_input.size(), 0);

    std::transform(test_input.begin(), test_input.end(), encoded_test_input.begin(),
                   [frac_bits](auto j) {
                     return MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
                   });
    encoded_x = encoded_test_input;
    // std::cout << "encoded test input size:  " << encoded_x.size() << "\n";
    // std::cout << "m:  " << m << "\n";

    int n0 = encoded_x.size() / m;
    enc_z1.assign(n1 * m, 0);

    enc_z1 = MOTION::matrix_multiply(n1, n0, m, enc_w1, encoded_x);
    std::transform(enc_z1.begin(), enc_z1.end(), enc_z1.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "z1 after multiplying w1 and x \n";
    // for (int i = 0; i < enc_z1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << "z1_encoded size : " << enc_z1.size() << "\n";
    // std::cout << "b1 size : " << enc_b1.size() << "\n";

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n1; j++) {
        enc_z1[i + m * j] = enc_z1[i + m * j] + enc_b1[j];
      }
    }

    // std::cout << "z1 after adding b1 \n";
    // for (int i = 0; i < enc_z1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << "z1_encoded size : " << enc_z1.size() << "\n";

    enc_a1.assign(enc_z1.size(), 0);

    enc_a1 = enc_z1;

    // activation_function(enc_a1, "relu", options);
    activation_function(enc_a1, "relu", options, *this);

    // std::cout << "a1 size : " << enc_a1.size() << "\n";

    // std::cout << "a1 after activation function: \n";
    // for (int i = 0; i < enc_a1.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_a1[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << m << "\n";
    // std::cout << enc_w2.size() << "\n";

    enc_z2.assign(n2 * m, 0);

    enc_z2 = MOTION::matrix_multiply(n2, n1, m, enc_w2, enc_a1);

    std::transform(enc_z2.begin(), enc_z2.end(), enc_z2.begin(), [frac_bits](auto j) {
      return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
    });

    // std::cout << "z2_encoded size : " << enc_z2.size() << "\n";

    // std::cout << "z2 after multiplying w2 and a1 \n";
    // for (int i = 0; i < enc_z2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    // std::cout << "b2 size : " << enc_b2.size() << "\n";

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n2; j++) {
        enc_z2[i + m * j] = enc_z2[i + m * j] + enc_b2[j];
      }
    }

    // std::cout << "z2 after adding b2 \n";
    // for (int i = 0; i < enc_z2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_z2[i], frac_bits)
    //             << " ";
    // }
    // std::cout << "\n\n";

    std::vector<float> dec_z2(n2 * m, 0);

    std::transform(enc_z2.begin(), enc_z2.end(), dec_z2.begin(), [frac_bits](auto j) {
      auto temp = MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
      return temp;
    });

    int size_single_z2 = dec_z2.size() / m;
    std::vector<float> dec_z2_transpose;
    dec_z2_transpose.assign(dec_z2.size(), 0);
    // std::cout << "decoded z2 transpose \n";
    int k;
    int count;
    int index = 0;
    for (int j = 0; j < m; j++) {
      count = 0;
      k = j;

      while (count != size_single_z2) {
        auto temp = dec_z2[k];
        dec_z2_transpose[index++] = temp;
        // std::cout << temp << " ";
        k = k + m;
        count++;
      }
      // std::cout << "\n";
    }
    /*********************************************************************************/
    // enc_a2.assign(enc_z2.size(), 0);
    // enc_a2 = enc_z2;

    // activation_function(enc_a2, "sigmoid", options, *this);

    // std::cout << "a2 size : " << enc_a2.size() << "\n";

    // std::cout << "a2 after activation function: \n";
    // for (int i = 0; i < enc_a2.size(); i++) {
    //   std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(enc_a2[i],
    //   frac_bits)
    //             << " ";
    // }
    // std::cout << "\n";

    // std::cout << "a2 transpose after activation function: \n";

    // std::vector<std::uint64_t> enc_a2_transpose;
    // std::vector<float> dec_a2_transpose;
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < options.classes; j++) {
    //     auto data = enc_a2[i + m * j];
    //     std::cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(data, frac_bits)
    //     << "
    //     ";
    //     // auto data_dec = MOTION::new_fixed_point::decode<std::uint64_t, float>(data,
    //     frac_bits);
    //     // enc_a2_transpose.push_back(data);
    //     // dec_a2_transpose.push_back(data_dec);
    //   }
    //   std::cout << "\n";
    // }

    // std::cout << " enc_a2 size : " << enc_a2.size() << "\n ";
    // std::cout << " test size : " << test_label.size() << "\n";

    /*********************************************************************************/

    // i - classes
    // m = test size
    std::vector<int> predictions;
    int m_sample = 0;
    while (m_sample != m) {
      int maxIndex = 0;
      int maxValue = dec_z2_transpose[m_sample * n2 + 0];
      for (int i = 1; i < n2; i++) {
        // Initialize variables to keep track of the maximum value and its index in each row
        if (dec_z2_transpose[m_sample * n2 + i] > maxValue) {
          maxValue = dec_z2_transpose[m_sample * n2 + i];
          maxIndex = i;
        }
      }
      predictions.push_back(maxIndex);
      m_sample++;
    }

    std::cout << "predictions size : " << predictions.size() << "\n";

    std::cout << "Predictions: ";
    for (int i : predictions) {
      std::cout << i << " ";
    }
    std::cout << "\nActual lab: ";
    for (auto i : test_label) {
      std::cout << i << " ";
    }

    int accuracy = 0;
    for (int i = 0; i < m; i++) {
      if (predictions[i] == (int)test_label[i]) {
        accuracy++;
      }
    }
    std::cout << "Accuracy = " << accuracy << "\n";
  }
};

std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("sample-file", po::value<std::string>()->required(), "csv file containing sample file names")
    ("sample-size", po::value<std::size_t>()->required(), "sample size")
    ("actual-label-file", po::value<std::string>()->required(), "csv file containing actual labels")
    ("fractional-bits", po::value<std::size_t>()->required(), "fractional bits")
    ("classes", po::value<std::size_t>()->required(), "classes")
    ("training-class", po::value<std::size_t>()->required(), "to be trained class")
;

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

  options.m = vm["sample-size"].as<std::size_t>();
  options.sample_file = vm["sample-file"].as<std::string>();
  options.actual_label_file = vm["actual-label-file"].as<std::string>();
  options.frac_bits = vm["fractional-bits"].as<std::size_t>();
  options.classes = vm["classes"].as<std::size_t>();
  options.training_class = vm["training-class"].as<std::size_t>();
  return options;
}


void activation_function(std::vector<std::uint64_t>& input , std::string function_type ,Options& options , const NeuralNetwork& network)
{  
    int frac_bits = options.frac_bits;
    std::uint64_t msb = (uint64_t)1 << 63;
  if(function_type=="relu")
  {
        std::transform(input.begin(), input.end(), input.begin(),
        [msb](std::uint64_t input_value) {
            if(msb == (msb & input_value))
            return static_cast<std::uint64_t>(0); 
            else 
            return input_value;
        });
  }

  else if(function_type=="sigmoid")
  {

    std::transform(input.begin(), input.end(), input.begin(),
        [msb,options](std::uint64_t input_value) {
      auto encoded_threshold = MOTION::new_fixed_point::encode<std::uint64_t, float>(0.5, options.frac_bits);
      auto neg_encoded_threshold =
      MOTION::new_fixed_point::encode<std::uint64_t, float>(-0.5, options.frac_bits);

    if (input_value & msb) {
    if (input_value <= neg_encoded_threshold) {
      return static_cast<std::uint64_t>(0); 
    } else {
      return input_value + encoded_threshold;
    }
    } else {
    if (input_value >= encoded_threshold) {
      return MOTION::new_fixed_point::encode<std::uint64_t, float>(1, options.frac_bits);
    } else {
      return input_value + encoded_threshold;
    }
    }
  });
  
}
}

int main(int argc, char* argv[]) {

  auto options = parse_program_options(argc, argv);

  int n0 = 784;
  int n1 = 20;
  int n2 = 10;
  int iterations = 3;
  float alpha = 0.005; 

  NeuralNetwork neuralNetwork(n0,n1,n2,*options);
  
  for(int e=0;e<1;e++)
  {
  for(int i=0;i<iterations;i++)
  {
  neuralNetwork.read_input(*options);
  neuralNetwork.encoded_inputs(*options);
  std::cout<<"\n*******iteration number**********  :"<<i+1<<"\n";
  std::cout<<"\n*******epoch number**********  :"<<e+1<<"\n";
  neuralNetwork.forward_propogation(*options,n1,n2,false); 
  neuralNetwork.cost_function(*options,n1,n2,n0);  
  neuralNetwork.backward_propogation(*options,alpha);
  }
  }
  neuralNetwork.print_weights_biases(*options);
  // neuralNetwork.test_accuracy(*options,n1,n2); 
}