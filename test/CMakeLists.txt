set(GTEST_SHUFFLE 1)

add_executable(
    detect-test
    main.cpp
    VisionInputTest.cpp
    ../app/VisionInput.cpp
    ../app/DataLoader.cpp
    DataLoaderTest.cpp
    ../app/DetectHuman.cpp
    DetectHumanTest.cpp
    ../app/TrainSVM.cpp
    TrainSVMTest.cpp
)

target_include_directories(detect-test PUBLIC ./ ../vendor/googletest/googletest/include ../vendor/googletest/googlemock/include  
                                           ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detect-test PUBLIC gtest gmock ${OpenCV_LIBS})
