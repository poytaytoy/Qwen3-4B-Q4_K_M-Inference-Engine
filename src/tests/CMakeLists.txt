find_package(Catch2 3 CONFIG REQUIRED)

add_executable(inference_tests
    TokenizerTests.cpp
    ChatFormatterTests.cpp
    EmbedLookupTests.cpp
    MatmulTests.cpp
    AttentionTests.cpp
)
target_link_libraries(inference_tests PRIVATE inference_core ggml-cpu Catch2::Catch2WithMain)

set(INFERENCE_TEST_MODEL "${MODEL_PATH}")
configure_file(TestModel.hpp.in TestModel.hpp @ONLY)
target_include_directories(inference_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

include(Catch)
catch_discover_tests(inference_tests PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
