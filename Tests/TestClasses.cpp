#if UNIT_TESTING

#include <CppUnitTest.h>
#include <Windows.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

// Test classes
// Expect all tests to be run in the same process, i.e. as if they where called
// by one master test method.
// Thus, later tests might fail simply because they encounter an invalid global state,
// for example a cuda device in which an assertion failed or the driver crashed etc.
// TODO it would be nice to have each test wrapped in a clean environment.
// This is possible to some extent using cudaDeviceReset and freeing resources etc.
// The test class is reconstructed separately for each test.
void tests();
void redirectStd(); void flushStd();
TEST_CLASS(MyTestClass) {
public:
    MyTestClass() {
        redirectStd();
    }

    ~MyTestClass() {
        flushStd();
    }
    TEST_METHOD(Tests) {
        tests();
    }
};

#endif