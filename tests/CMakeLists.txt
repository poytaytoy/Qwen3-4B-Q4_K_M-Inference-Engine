find_package(Catch2 3 CONFIG REQUIRED)

add_executable(inference_tests
    TokenizerTests.cpp
    ChatFormatterTests.cpp
    EmbedLookupTests.cpp
)
target_link_libraries(inference_tests PRIVATE inference_core Catch2::Catch2WithMain)

set(INFERENCE_TEST_MODEL "${PROJECT_SOURCE_DIR}/models/Qwen3-4B-Q4_K_M.gguf"
    CACHE FILEPATH "GGUF model used by functional tests")
get_filename_component(INFERENCE_TEST_MODEL "${INFERENCE_TEST_MODEL}" ABSOLUTE
    BASE_DIR "${PROJECT_SOURCE_DIR}")
configure_file(TestModel.hpp.in TestModel.hpp @ONLY)
target_include_directories(inference_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

include(Catch)
catch_discover_tests(inference_tests PROPERTIES RUN_SERIAL TRUE TIMEOUT 180)
