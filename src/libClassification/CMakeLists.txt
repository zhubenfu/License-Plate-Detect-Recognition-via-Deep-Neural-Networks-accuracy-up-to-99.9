file(GLOB_RECURSE libclassification_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(libClassification SHARED ${libclassification_srcs})
caffe_default_properties(libClassification)
target_link_libraries(libClassification ${Caffe_LINK})

##target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)


#add_library(gtest_main gtest_main.cc)
#target_link_libraries(gtest_main gtest)
