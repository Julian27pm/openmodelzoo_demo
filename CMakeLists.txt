project(demo)
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SRCS
  src/main.cpp
  src/flags_handler.cpp
  src/detector.cpp
  src/pool.cpp
  src/worker.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

find_package(OpenCV 4 REQUIRED)
find_package(OpenVINO REQUIRED)

target_link_libraries(${PROJECT_NAME} 
  PRIVATE 
    ${OpenCV_LIBS}
    openvino::runtime
)
target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
