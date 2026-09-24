# Builds cpp_smoke.cpp against an installed Brook SDK: cmake -DCMAKE_PREFIX_PATH=<prefix>.
cmake_minimum_required(VERSION 3.24)
project(BrookConsumer LANGUAGES CXX)
find_package(Brook CONFIG REQUIRED)
add_executable(consumer ../cpp_smoke.cpp)
target_link_libraries(consumer PRIVATE Brook::brook)
