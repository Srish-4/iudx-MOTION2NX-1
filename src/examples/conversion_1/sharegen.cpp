#include <bits/stdc++.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <boost/archive/text_oarchive.hpp>
#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/string.hpp>
#include "utility/fixed_point.h"

template <typename E>
std::uint64_t blah(E& engine) {
  std::uniform_int_distribution<unsigned long long> dis(std::numeric_limits<std::uint64_t>::min(),
                                                        std::numeric_limits<std::uint64_t>::max());
  return dis(engine);
}

void generate_random_numbers() {
  std::random_device rd;   // Seed for the random number engine
  std::mt19937 gen(rd());  // Mersenne Twister random number generator

  int minNumber = 0;
  int maxNumber = 9;
  std::uniform_int_distribution<int> distribution(minNumber, maxNumber);
  // std::uniform_real_distribution<float> distribution(minNumber, maxNumber);
  std::ofstream file;
  std::string home_dir = getenv("BASE_DIR");
  std::string path = home_dir + "/build_debwithrelinfo_gcc/random_numbers";
  // std::string path = home_dir + "/data/ModelProvider/W1_dec.csv";
  file.open(path, std::ios_base::out);
  for (int i = 0; i < 64; i++) {
    int random_number = distribution(gen);
    // file << random_number << "\n";
    std::cout << random_number << "\n";
  }
  std::cout << "\n";
  file.close();
}

void generate_numbers() {
  std::ofstream file;
  std::string home_dir = getenv("BASE_DIR");
  // std::string path = home_dir + "/build_debwithrelinfo_gcc/test_numbers";
  std::string path = home_dir + "/build_debwithrelinfo_gcc/random_numbers";
  file.open(path, std::ios_base::out);

  for (float value = 1; value <= 784 * 10; value += 1) {
    file << 0 << "\n";
  }
  file.close();
}

void share_generation(int rows, int columns, std::vector<float> data) {
  static int k = 0;

  auto p = std::filesystem::current_path();
  auto q = std::filesystem::current_path();
  // p += "/server0/outputshare_0";
  // q += "/server1/outputshare_1";
  // p += "/SharesForW2T0";
  // q += "/SharesForW2T1";

  k += 1;
  // p += "/server0/Image_shares/Sample_shares" + std::to_string(k);
  // q += "/server1/Image_shares/Sample_shares" + std::to_string(k);
  p += "/server0/Image_shares/Actual_all_labels";
  q += "/server1/Image_shares/Actual_all_labels";
  // p += "/server0/Image_shares/Theta_all_labels";
  // q += "/server1/Image_shares/Theta_all_labels";
  std::cout << p << " " << q << "\n";
  std::ofstream file1, file2;
  file1.open(p, std::ios_base::out);
  if (!file1) {
    std::cerr << " Error in reading file 1\n";
  }
  file2.open(q, std::ios_base::out);
  if (!file2) {
    std::cerr << " Error in reading file 2\n";
  }
  if (file1.is_open()) {
    file1 << rows;
    file1 << " ";

    file1 << columns;
    file1 << "\n";
  }
  if (file2.is_open()) {
    file2 << rows;
    file2 << " ";

    file2 << columns;
    file2 << "\n";
  }

  int size_of_data = data.size();
  std::cout << size_of_data << "\n";
  ;
  std::cout << "size of data:" << size_of_data << "\n";
  // Now that we have data, need to generate the shares
  for (int i = 0; i < size_of_data; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uint64_t del0 = blah(gen);
    std::uint64_t del1 = blah(gen);

    std::uint64_t Del = del0 + del1 + MOTION::fixed_point::encode<uint64_t, float>(data[i], 13);

    //////////Writing shares for server 0//////////////
    if (file1.is_open()) {
      file1 << Del;
      file1 << " ";

      file1 << del0;
      file1 << "\n";
    }

    //////////Writing shares for server 1//////////////
    if (file2.is_open()) {
      file2 << Del;
      file2 << " ";

      file2 << del1;
      file2 << "\n";
    }
  }

  file1.close();
  file2.close();
}

int main() {
  // generate_random_numbers();

  // generate_numbers();
  int r = 10;
  int c = 1;
  int m = 1;

  std::ifstream file1;
  std::vector<float> input;
  std::string home_dir = getenv("BASE_DIR");

  // std::string path = home_dir + "/build_debwithrelinfo_gcc/random_numbers";
  // std::string path = home_dir + "/build_debwithrelinfo_gcc/test_numbers";

  // // IMAGE TRAINING DATA share generation
  // std::string path = home_dir + "/data/ImageProvider/sample_data/";
  // std::string input_path;
  // std::string path1 = home_dir + "/data/ImageProvider/sample_data/" + std::to_string(m) +
  // "_folder"; std::string path2 = home_dir + "/data/ImageProvider/sample_data/" +
  // std::to_string(m) + "_images";

  // std::ifstream file_1, file_2;
  // file_1.open(path1);
  // int temp;
  // std::vector<int> folder;
  // std::vector<int> images;
  // while (file_1) {
  //   file_1 >> temp;
  //   std::cout << temp << " ";
  //   folder.push_back(temp);
  // }
  // std::cout << "\n";

  // file_2.open(path2);
  // while (file_2) {
  //   file_2 >> temp;
  //   std::cout << temp << " ";
  //   images.push_back(temp);
  // }
  // std::cout << "\n";

  // for (int i = 0; i < m; i++) {
  //   input_path = path + "images_folder" + std::to_string(folder[i]) + "/X" +
  //                std::to_string(images[i]) + ".csv";
  //   std::cout << input_path << "\n";

  //   file1.open(input_path);
  //   std::string str;
  //   std::vector<float> input_vec;
  //   while (std::getline(file1, str)) {
  //     std::stringstream obj(str);
  //     std::string temp;
  //     while (std::getline(obj, temp, ',')) {
  //       auto input = std::stof(temp);
  //       input_vec.push_back(input);
  //     }
  //   }
  //   share_generation(r, c, input_vec);
  //   input_vec.clear();
  //   file1.close();
  // }

  // MODEL share generation
  // std::string path = home_dir + "/data/ModelProvider/theta_dec";
  // file1.open(path);
  // std::string str;
  // while (std::getline(file1, str)) {
  //   std::stringstream obj(str);
  //   std::string temp;
  //   while (std::getline(obj, temp, '\n')) {
  //     auto x = std::stof(temp);
  //     input.push_back(x);
  //   }
  // }
  // std::cout << "input size: " << input.size() << "\n";
  // share_generation(r, c, input);
  // input.clear();
  // file1.close();

  // ACTUAL LABEL SHARE GENERATION
  // std::string path = home_dir + "/data/ImageProvider/sample_data/" + std::to_string(m) +
  // "_folder"; std::vector<float> actual_label; std::vector<float> actual_label_test;
  // actual_label.assign(10 * m, 0);
  // file1.open(path);
  // while (!file1.eof()) {
  //   float data;
  //   file1 >> data;
  //   std::cout << data << " ";
  //   actual_label_test.push_back(data);
  // }
  // std::cout << "\n";
  // file1.close();

  // int t = 0;
  // std::cout << "testing: \n";
  // for (int i = 0; i < m; i++) {
  //   t = i * 10 + actual_label_test[i];
  //   std::cout << t << " ";
  //   actual_label[t] = 1;
  // }
  // std::cout << "\n";

  // std::cout << "Actual label : \n";
  // for (int i = 0; i < actual_label.size(); i++) {
  //   std::cout << actual_label[i] << " ";
  //   if ((i + 1) % 10 == 0) std::cout << "\n";
  // }

  // std::cout << "end\n";
  // //***************************************************************************

  // std::vector<float> actual_label_transpose;
  // for (int i = 0; i < 10; ++i) {
  //   for (int j = 0; j < m; j++) {
  //     auto data = actual_label[j * 10 + i];
  //     std::cout << data << " ";
  //     actual_label_transpose.push_back(data);
  //   }
  //   std::cout << "\n";
  // }
  // share_generation(r, c, actual_label_transpose);

  // Splitwise actual label generation
  // std::string path = home_dir + "/data/ImageProvider/sample_data/" + std::to_string(8) +
  // "_folder"; std::vector<float> actual_label; std::vector<float> actual_label_test;
  // actual_label.assign(10 * m, 0);
  // file1.open(path);
  // while (file1) {
  //   float data;
  //   file1 >> data;
  //   std::cout << data << " ";
  //   actual_label_test.push_back(data);
  // }
  // std::cout << "\n";
  // file1.close();

  // auto x = actual_label_test[6];
  // actual_label[x] = 1;

  // std::cout << "Actual label : \n";
  // for (int i = 0; i < actual_label.size(); i++) {
  //   std::cout << actual_label[i] << " ";
  // }
  // std::cout << "\n";

  // share_generation(r, c, actual_label);

  // ENCODE MODEL SHARES
  // std::string path = home_dir + "/data/ModelProvider/B2_dec.csv";
  // file1.open(path);
  // std::string str;
  // while (std::getline(file1, str)) {
  //   std::stringstream obj(str);
  //   std::string temp;
  //   while (std::getline(obj, temp, '\n')) {
  //     auto x = std::stof(temp);
  //     input.push_back(x);
  //   }
  // }

  // std::string path1 = home_dir + "/data/ModelProvider/B2_enc.csv";
  // std::ofstream file;
  // file.open(path1, std::ios_base::out);

  // for (int i = 0; i < input.size(); i++) {
  //   auto x = MOTION::fixed_point::encode<uint64_t, float>(input[i], 13);
  //   file << x << "\n";
  // }
  // file.close();

  // Calculation of root mean square ....
  // std::vector<double> error;
  // std::vector<double> arr1, arr2;
  // std::ifstream file1, file2;
  // std::string path1 = home_dir + "/build_debwithrelinfo_gcc/Theta64_clear";
  // std::string path2 = home_dir + "/build_debwithrelinfo_gcc/Theta64_smpc";

  // file1.open(path1);
  // file2.open(path2);
  // double x;
  // while (file1) {
  //   file1 >> x;
  //   arr1.push_back(x);
  // }

  // while (file2) {
  //   file2 >> x;
  //   arr2.push_back(x);
  // }

  // double sum = 0;
  // std::cout << arr1.size() << "\n";
  // for (int i = 0; i < arr1.size(); i++) {
  //   auto y = arr2[i] - arr1[i];
  //   auto d = pow(y, 2);
  //   // std::cout << d << " ";
  //   sum += d;
  // }
  // std::cout << "\n";
  // std::cout << "sum: " << sum << "\n";
  // double mean = sum / 7840;
  // std::cout << "mean: " << mean << "\n";
  // double k = pow(mean, 0.5);

  // std::cout << "rms: " << k << "\n";

  // std::cout << "\n";
}
