cmake_minimum_required(VERSION 2.8)
project(srv_ecto_vision)
find_package(catkin REQUIRED)
catkin_stack()

set(${PROJECT_NAME}_GITHUB_URL https://github.com/srv/srv_ecto_vision)

if(NOT CMAKE_BUILD_TYPE)#Only do this the on the first run, if the build type hasn't been set prior
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

add_definitions(-Wno-strict-aliasing -Wall -Werror)

find_package(OpenCV REQUIRED core imgproc highgui calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV headers: ${OpenCV_INCLUDE_DIRS}")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(ecto REQUIRED)
find_package(Boost COMPONENTS serialization filesystem system REQUIRED)

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

rosbuild_lite_init()
set_ecto_install_package_name(srv_ecto_vision)

# configure and install Python
catkin_python_setup()

# install the include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(cells)
#add_subdirectory(src)

#option(BUILD_TEST "Build tests" ON)
#if(BUILD_TEST)
#  enable_testing()
#    add_subdirectory(test)
#  add_subdirectory(samples)
#endif()

catkin_project(srv_ecto_vision
 INCLUDE_DIRS
 include
)

#catkin_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc doc)
