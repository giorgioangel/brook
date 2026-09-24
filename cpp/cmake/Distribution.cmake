# Shared wheel/SDK install policy. Runtime libraries stay independently replaceable.
option(BROOK_BUNDLE_CUDA_RUNTIME "Bundle the unmodified CUDA runtime beside Brook" ON)

set(BROOK_PYTHON_FILES
  __init__.py py.typed _core.pyi intake.py packed.py pool.py post.py device.py device_post.py
  trace.py streaming.py profile.py preamble.py
  _runtime.py _intake.py _helpers.py _packed.py batch.py
  _device_post.py _pool.py)
set(BROOK_PYTHON_CONTRIB_FILES __init__.py igneous.py _igneous.py torch.py)

# Use the final relative RPATH in both build and install trees so wheel installs
# can copy just the SONAME file, avoiding three copies of a large CUDA DSO.
set_target_properties(brook PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
target_compile_features(brook PUBLIC cxx_std_17)

# Release builds link Brook's own shared objects without symbol tables (smaller, and free of the
# temporary file names nvcc leaves in them). The copied CUDA runtime is never altered.
set(_brook_strip "$<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:LINKER:-s>")
target_link_options(brook PRIVATE "${_brook_strip}")
target_link_options(brook_xs3d PRIVATE "${_brook_strip}")
if(BROOK_BUILD_PYTHON)
  target_link_options(_core PRIVATE "${_brook_strip}")
endif()

if(BROOK_BUNDLE_CUDA_RUNTIME)
  get_target_property(_brook_cudart CUDA::cudart IMPORTED_LOCATION)
  if(NOT _brook_cudart)
    message(FATAL_ERROR "CUDA::cudart has no runtime library location")
  endif()
  file(REAL_PATH "${_brook_cudart}" BROOK_CUDART_REAL)
  if(NOT CMAKE_READELF)
    find_program(CMAKE_READELF readelf REQUIRED)
  endif()
  execute_process(COMMAND "${CMAKE_READELF}" -d "${BROOK_CUDART_REAL}"
    OUTPUT_VARIABLE _brook_dynamic RESULT_VARIABLE _brook_readelf_status)
  if(NOT _brook_readelf_status EQUAL 0 OR NOT _brook_dynamic MATCHES "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
    message(FATAL_ERROR "Cannot determine CUDA runtime SONAME")
  endif()
  set(BROOK_CUDART_SONAME "${CMAKE_MATCH_1}")
  # -DBROOK_CUDA_EULA=<file> names the EULA and skips the search; each find_file below runs
  # only while none has been found.
  set(_brook_toolkit_roots "${CUDAToolkit_LIBRARY_ROOT}" "${CUDAToolkit_ROOT_DIR}" "${CUDAToolkit_BIN_DIR}/..")
  list(FILTER _brook_toolkit_roots EXCLUDE REGEX "^$")
  list(REMOVE_DUPLICATES _brook_toolkit_roots)
  find_file(BROOK_CUDA_EULA NAMES EULA.txt HINTS ${_brook_toolkit_roots} NO_DEFAULT_PATH)
  # Toolkits installed from distribution packages keep the complete CUDA EULA with the runtime
  # package rather than at the toolkit root: NVIDIA's Debian packages (cuda-cudart-X-Y), Debian and
  # Ubuntu's own packages (libcudartX, as in Lambda Stack), and NVIDIA's RPM packages (as in the
  # manylinux build).
  set(_brook_cudart_package "cuda-cudart-${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}")
  find_file(BROOK_CUDA_EULA NAMES copyright HINTS "/usr/share/doc/${_brook_cudart_package}" NO_DEFAULT_PATH)
  # libcudartX is named by the major version only, so its EULA is taken only for the runtime that
  # package installs (under /usr/lib), not for a runtime from another toolkit.
  if(BROOK_CUDART_REAL MATCHES "^/usr/lib/")
    find_file(BROOK_CUDA_EULA NAMES copyright
      HINTS "/usr/share/doc/libcudart${CUDAToolkit_VERSION_MAJOR}" NO_DEFAULT_PATH)
  endif()
  find_file(BROOK_CUDA_EULA NAMES LICENSE
    HINTS "/usr/share/licenses/${_brook_cudart_package}"
    NO_DEFAULT_PATH)
  if(NOT BROOK_CUDA_EULA)
    list(JOIN _brook_toolkit_roots ", " _brook_roots_text)
    message(FATAL_ERROR "Cannot find the CUDA EULA to ship with the bundled CUDA runtime "
      "(${BROOK_CUDART_REAL}). Searched EULA.txt in ${_brook_roots_text}, "
      "/usr/share/doc/${_brook_cudart_package}/copyright, "
      "/usr/share/doc/libcudart${CUDAToolkit_VERSION_MAJOR}/copyright (for a runtime under /usr/lib) and "
      "/usr/share/licenses/${_brook_cudart_package}/LICENSE. "
      "Pass -DBROOK_CUDA_EULA=<file>, or -DBROOK_BUNDLE_CUDA_RUNTIME=OFF.")
  endif()
  message(STATUS "Brook bundles ${BROOK_CUDART_REAL} as ${BROOK_CUDART_SONAME}, with the CUDA EULA from ${BROOK_CUDA_EULA}")
  # This is a byte-for-byte copy, with no strip/RPATH alteration to NVIDIA code.
  configure_file("${BROOK_CUDART_REAL}" "${CMAKE_CURRENT_BINARY_DIR}/${BROOK_CUDART_SONAME}" COPYONLY)
  install(FILES "${BROOK_CUDART_REAL}" DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RENAME "${BROOK_CUDART_SONAME}" COMPONENT SDK)
  install(FILES "${BROOK_CUDA_EULA}" DESTINATION "${CMAKE_INSTALL_DATADIR}/brook/licenses/nvidia"
    RENAME CUDA-EULA.txt COMPONENT SDK)
endif()

# Include corresponding sources and the independently rebuildable LGPL bridge.
function(brook_install_sources destination component)
  install(FILES CMakeLists.txt pyproject.toml README.md LICENSE NOTICE
    DESTINATION "${destination}" COMPONENT "${component}")
  # The scripts that build and publish the release wheels, and the locked build tools they install.
  install(PROGRAMS tools/build_manylinux.sh DESTINATION "${destination}/tools" COMPONENT "${component}")
  install(FILES uv.lock DESTINATION "${destination}" COMPONENT "${component}")
  install(FILES .github/workflows/release.yml DESTINATION "${destination}/.github/workflows" COMPONENT "${component}")
  install(DIRECTORY cpp/ DESTINATION "${destination}/cpp" COMPONENT "${component}")
  install(DIRECTORY docs/ DESTINATION "${destination}/docs" COMPONENT "${component}")
  install(DIRECTORY examples/ DESTINATION "${destination}/examples" COMPONENT "${component}"
    FILES_MATCHING PATTERN "*.py")
  install(DIRECTORY tests/python/ DESTINATION "${destination}/tests/python" COMPONENT "${component}"
    FILES_MATCHING PATTERN "*.py")
  foreach(_name IN LISTS BROOK_PYTHON_FILES)
    install(FILES "src/brook/${_name}" DESTINATION "${destination}/src/brook" COMPONENT "${component}")
  endforeach()
  foreach(_name IN LISTS BROOK_PYTHON_CONTRIB_FILES)
    install(FILES "src/brook/contrib/${_name}" DESTINATION "${destination}/src/brook/contrib" COMPONENT "${component}")
  endforeach()
endfunction()

brook_install_sources("${CMAKE_INSTALL_DATADIR}/brook/source" SDK)
install(FILES LICENSE NOTICE DESTINATION "${CMAKE_INSTALL_DATADIR}/brook" COMPONENT SDK)

if(BROOK_BUILD_PYTHON)
  set_target_properties(_core PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN/.libs")
  install(TARGETS _core LIBRARY DESTINATION brook COMPONENT Python)
  install(FILES "$<TARGET_FILE:brook>" DESTINATION brook/.libs
    RENAME "$<TARGET_SONAME_FILE_NAME:brook>" COMPONENT Python)
  install(FILES "$<TARGET_FILE:brook_xs3d>" DESTINATION brook/.libs
    RENAME "$<TARGET_SONAME_FILE_NAME:brook_xs3d>" COMPONENT Python)
  foreach(_name IN LISTS BROOK_PYTHON_FILES)
    install(FILES "src/brook/${_name}" DESTINATION brook COMPONENT Python)
  endforeach()
  foreach(_name IN LISTS BROOK_PYTHON_CONTRIB_FILES)
    install(FILES "src/brook/contrib/${_name}" DESTINATION brook/contrib COMPONENT Python)
  endforeach()
  install(DIRECTORY cpp/licenses/ DESTINATION brook/licenses COMPONENT Python)
  install(FILES LICENSE NOTICE DESTINATION brook COMPONENT Python)
  install(DIRECTORY cpp/third_party/xs3d/ DESTINATION brook/_third_party/xs3d COMPONENT Python)
  brook_install_sources("brook/_source" Python)
  if(BROOK_BUNDLE_CUDA_RUNTIME)
    install(FILES "${BROOK_CUDART_REAL}" DESTINATION brook/.libs
      RENAME "${BROOK_CUDART_SONAME}" COMPONENT Python)
    install(FILES "${BROOK_CUDA_EULA}" DESTINATION brook/licenses/nvidia
      RENAME CUDA-EULA.txt COMPONENT Python)
  endif()
endif()
