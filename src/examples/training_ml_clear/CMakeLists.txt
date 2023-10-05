add_executable(logistic_regression logistic_regression.cpp)
add_executable(logistic_regression_10 logistic_regression_10.cpp)
add_executable(training_neuralnet training_neuralnet.cpp)

find_package(Boost COMPONENTS json log program_options REQUIRED)


target_compile_features(logistic_regression PRIVATE cxx_std_20)
target_compile_features(logistic_regression_10 PRIVATE cxx_std_20)
target_compile_features(training_neuralnet PRIVATE cxx_std_20)

target_link_libraries(logistic_regression
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

target_link_libraries(training_neuralnet
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)
