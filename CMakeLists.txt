cmake_minimum_required(VERSION 3.0.2)
project(load_pcd)

# SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES load_pcd
#  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

find_package(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

find_package(OpenCV REQUIRED)

add_executable(pcl_seam src/pcl_seam.cpp)
target_link_libraries(pcl_seam ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
