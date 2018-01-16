cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.0 REQUIRED)
if(OpenCV_FOUND)
    message("--- Found OpenCV.")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
	message("--- Could not find OpenCV.")
endif()

add_executable(result main.cpp)
target_link_libraries(result ${OpenCV_LIBS})
