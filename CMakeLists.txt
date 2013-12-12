cmake_minimum_required(VERSION 2.8.3)
project(facetraking)

set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")

find_package(OpenCV REQUIRED COMPONENTS core video imgproc objdetect highgui contrib )

find_package(PkgConfig)
pkg_search_module(log4cxx REQUIRED liblog4cxx)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${log4cxx_INCLUDE_DIRS}
)

add_definitions(-std=c++11)
add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

add_library(facetracking SHARED
            src/facetracking.cpp
            src/human.cpp
            src/detection.cpp 
            src/recognition.cpp)

add_executable(facetracker 
                src/main.cpp)

target_link_libraries(facetracking
   ${OpenCV_LIBRARIES}
)

target_link_libraries(facetracker
   facetracking
   ${OpenCV_LIBRARIES}
   ${log4cxx_LIBRARIES}
)

install(TARGETS facetracking facetracker
   LIBRARY DESTINATION lib 
   RUNTIME DESTINATION bin 
)

file(
    GLOB_RECURSE
    facetracking_headers
    include/*
)

install(FILES
    ${facetraking_headers}
    DESTINATION include/facetracking
)

file(
    GLOB_RECURSE
    facetracking_data
    share/facetracking/*
)

install(FILES
    ${facetracking_data}
    DESTINATION share/facetracking
)


option (WITH_TESTS "build tests" OFF)

if (WITH_TESTS)
    #enable_testing()
    add_subdirectory(testing)
endif()
