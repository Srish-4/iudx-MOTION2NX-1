add_executable(machine_learning_1l_clear_5sigmoid machine_learning_1l_clear_5sigmoid.cpp)
add_executable(machine_learning_1l_smpc_5sigmoid machine_learning_1l_smpc_5sigmoid.cpp)

find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(machine_learning_1l_clear_5sigmoid PRIVATE cxx_std_20)
target_compile_features(machine_learning_1l_smpc_5sigmoid PRIVATE cxx_std_20)

target_link_libraries(machine_learning_1l_clear_5sigmoid
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(machine_learning_1l_smpc_5sigmoid
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)
