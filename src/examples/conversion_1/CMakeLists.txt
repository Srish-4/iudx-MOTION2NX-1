add_executable(readfile readfile.cpp)
add_executable(appendfile appendfile.cpp)
add_executable(machine_learning machine_learning.cpp)
add_executable(machine_learning_1l_clear machine_learning_1l_clear.cpp)
add_executable(sharegen sharegen.cpp)
add_executable(matrixmultiply_test matrixmultiply_test.cpp)
add_executable(training_neuralnet_clear training_neuralnet_clear.cpp)
add_executable(training_neuralnet_smpc training_neuralnet_smpc.cpp)
add_executable(machine_learning_1l_smpc machine_learning_1l_smpc.cpp)
add_executable(cnn cnn.cpp)
add_executable(test_maxpool_op test_maxpool_op.cpp)


find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(readfile PRIVATE cxx_std_20)
target_compile_features(appendfile PRIVATE cxx_std_20)
target_compile_features(machine_learning PRIVATE cxx_std_20)
target_compile_features(machine_learning_1l_clear PRIVATE cxx_std_20)
target_compile_features(sharegen PRIVATE cxx_std_20)
target_compile_features(matrixmultiply_test PRIVATE cxx_std_20)
target_compile_features(training_neuralnet_clear PRIVATE cxx_std_20)
target_compile_features(training_neuralnet_smpc PRIVATE cxx_std_20)
target_compile_features(machine_learning_1l_smpc PRIVATE cxx_std_20)
target_compile_features(cnn PRIVATE cxx_std_20)
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

target_link_libraries(machine_learning
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(machine_learning_1l_clear
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

target_link_libraries(training_neuralnet_clear
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(training_neuralnet_smpc
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(machine_learning_1l_smpc
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)


target_link_libraries(cnn
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


