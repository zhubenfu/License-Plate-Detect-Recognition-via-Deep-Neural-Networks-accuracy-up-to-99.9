#file(GLOB_RECURSE ocr_test_src "${PROJECT_SOURCE_DIR}/ocr_test/*.cpp")
#message(${ocr_test_src})
#add_executable(ocr_test ${ocr_test_src})


add_executable(ocr_test ocr_test.cpp bktree.cpp ctcpp_entrypoint.cpp levenshtein.cpp)
##caffe_default_properties(ocr_test)
##add_dependencies(ocr_test libClassification caffe)
target_link_libraries(ocr_test ${Caffe_LINK} libClassification /usr/lib/x86_64-linux-gnu/libomp.so ${OpenCV_LIBS})

##target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)


#add_library(gtest_main gtest_main.cc)
#target_link_libraries(gtest_main gtest)
