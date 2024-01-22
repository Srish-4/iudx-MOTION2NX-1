add_executable(server1 server1.cpp)
add_executable(server0 server0.cpp)
add_executable(server2 server2.cpp)

add_executable(server1_cnn server1_cnn.cpp)
add_executable(server0_cnn server0_cnn.cpp)
add_executable(server2_cnn server2_cnn.cpp)


find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(server1 PRIVATE cxx_std_20)
target_compile_features(server0 PRIVATE cxx_std_20)
target_compile_features(server2 PRIVATE cxx_std_20)

target_compile_features(server1_cnn PRIVATE cxx_std_20)
target_compile_features(server0_cnn PRIVATE cxx_std_20)
target_compile_features(server2_cnn PRIVATE cxx_std_20)




target_link_libraries(server0
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)


target_link_libraries(server1
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(server2
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(server0_cnn
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)


target_link_libraries(server1_cnn
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(server2_cnn
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)


