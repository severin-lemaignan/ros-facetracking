
MACRO(declare_test)
    cmake_parse_arguments(declare_test "NEEDS_DATA" "TESTNAME" "CV_MODULES" ${ARGN} )

	file(GLOB ${declare_test_TESTNAME}_source_files ${declare_test_TESTNAME}*.cpp)
	add_executable(${declare_test_TESTNAME} ${${declare_test_TESTNAME}_source_files})

	include_directories(${CMAKE_SOURCE_DIR}/include)
	target_link_libraries(${declare_test_TESTNAME} facetracking)

    target_link_libraries(${declare_test_TESTNAME} ${OpenCV_LIBS} ${log4cxx_LIBRARIES} )

	add_test(${declare_test_TESTNAME} ${declare_test_TESTNAME})

	if (${declare_test_NEEDS_DATA})
		set_property(TEST ${declare_test_TESTNAME} PROPERTY ENVIRONMENT "OPENCV_TEST_DATA_PATH=${TEST_DATA}")
	endif()
endMACRO()

add_definitions(-std=c++11)

declare_test(TESTNAME tracking NEEDS_DATA)

add_executable(annotator 
               annotator.cpp)

target_link_libraries(annotator
   ${OpenCV_LIBRARIES}
)



