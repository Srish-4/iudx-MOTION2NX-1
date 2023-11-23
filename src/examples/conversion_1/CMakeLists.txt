add_executable(readfile readfile.cpp)
add_executable(appendfile appendfile.cpp)
add_executable(sharegen sharegen.cpp)
add_executable(matrixmultiply_test matrixmultiply_test.cpp)
add_executable(logistic_regression_10 logistic_regression_10.cpp)
add_executable(training_all_labels training_all_labels.cpp)
add_executable(training_neuralnet training_neuralnet.cpp)
add_executable(test_for_clear_nn2 test_for_clear_nn2.cpp)
add_executable(generation_of_random_wb generation_of_random_wb.cpp)
add_executable(test_maxpool_op test_maxpool_op.cpp)



find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(readfile PRIVATE cxx_std_20)
target_compile_features(appendfile PRIVATE cxx_std_20)
target_compile_features(sharegen PRIVATE cxx_std_20)
target_compile_features(matrixmultiply_test PRIVATE cxx_std_20)
target_compile_features(logistic_regression_10 PRIVATE cxx_std_20)
target_compile_features(training_all_labels PRIVATE cxx_std_20)
target_compile_features(training_neuralnet PRIVATE cxx_std_20)
target_compile_features(test_for_clear_nn2 PRIVATE cxx_std_20)
target_compile_features(generation_of_random_wb PRIVATE cxx_std_20)
target_compile_features(test_maxpool_op PRIVATE cxx_std_20)

target_link_libraries(readfile
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(appendfile
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(sharegen
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(matrixmultiply_test
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(logistic_regression_10
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(training_all_labels
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(training_neuralnet
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)


target_link_libraries(test_for_clear_nn2
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(generation_of_random_wb
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(test_maxpool_op
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)





