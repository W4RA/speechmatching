cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(acoustic acoustic.cpp)

find_package(flashlight CONFIG REQUIRED)

target_link_libraries(acoustic PRIVATE
    flashlight::flashlight
    flashlight::flashlight-app-asr
    flashlight::fl-libraries
)

