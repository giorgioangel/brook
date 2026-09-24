cmake_minimum_required(VERSION 3.16)
project(brook_xs3d LANGUAGES CXX)
add_library(brook_xs3d SHARED bridge.cpp)
target_compile_features(brook_xs3d PRIVATE cxx_std_17)
target_compile_options(brook_xs3d PRIVATE -ffp-contract=off)
set_target_properties(brook_xs3d PROPERTIES CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON VERSION 1.13.0 SOVERSION 1)
