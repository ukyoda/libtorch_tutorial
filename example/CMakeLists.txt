cmake_minimum_required(VERSION 3.10)

project(libtorch_tutorial)


if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "python3")
    endif()
    
    # Pytorchのライブラリのcmake_prefixを取得
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')" OUTPUT_VARIABLE pytorch_path)

    list(APPEND CMAKE_PREFIX_PATH "${pytorch_path}")
    
# libtorch find
find_package(Torch REQUIRED)

# OpenCV find
find_package(OpenCV REQUIRED)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ${TORCH_CXX_FLAGS}")

# Include Pathセット
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
)

add_executable(example src/main.cpp)
target_link_libraries(example ${OpenCV_LIBS} ${TORCH_LIBRARIES})