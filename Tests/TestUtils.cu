// For CUDADefines
#include <string> // wstring #include <locale> // wstring_convert
#include <codecvt>
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <string>
#include <regex>
#include <iterator>

#if UNIT_TESTING
#include "CppUnitTest.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
#endif

std::wstring _errmsg;

std::string editCudaError(std::string s) {
    // turn J:/Masterarbeit/SparseSpatialGrid/tests.cu:31: block: [0,0,0], thread: [0,0,0] Assertion `false` failed.
    // into J:/Masterarbeit/SparseSpatialGrid/tests.cu(31) : block: [0,0,0], thread: [0,0,0] Assertion `false` failed.
    // which is the syntax required to enable visual studio's jump-to-line for arbitrary output
    // "<filename>(<linenum>) : "
    std::regex e("(^[^\:]*\:[^\:]*)\:([0-9]*)\:(.*)[\r]*$");
    return std::regex_replace(s, e, "$1($2) :$3");
}

std::string readFile(std::string fn) {
    std::ifstream t(fn);
    return std::string(std::istreambuf_iterator<char>(t),
        std::istreambuf_iterator<char>());
}

#define stdoutfile "stdout.txt"
#define stderrfile "stderr.txt"
void redirectStd() {
    // These will be in x64\Debug
    freopen(stdoutfile, "w", stdout);
    freopen(stderrfile, "w", stderr);
}

#include <stdio.h>

#include <windows.h>
#include <tchar.h>
#include <stdio.h>

int fileExists(TCHAR * file)
{
    WIN32_FIND_DATA FindFileData;
    HANDLE handle = FindFirstFile(file, &FindFileData);
    int found = handle != INVALID_HANDLE_VALUE;
    if (found)
    {
        //FindClose(&handle); this will crash
        FindClose(handle);
    }
    return found;
}

void flushStd() {
    if (!fileExists(stdoutfile)) return;
    // unlock stdout.txt
    ::fflush(stdout);
    ::fflush(stderr);
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
    std::string s = (

        "<<< " stdoutfile " >>>\n" + readFile(stdoutfile) +
        "<<< " stderrfile " >>>\n" + editCudaError(readFile(stderrfile))

        );
#if UNIT_TESTING
    Logger::WriteMessage(
        s.c_str());
#endif
    OutputDebugStringA(s.c_str());
    remove(stdoutfile);
    remove(stderrfile);
}
#include <windows.h>
void logger(std::string s) {
    std::string narrow = s;
    _errmsg = std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(narrow);

#if UNIT_TESTING
    Logger::WriteMessage(_errmsg.c_str());
#endif
    flushStd();

}

bool assertImpl(bool f, const char* const expr, const char* const file, int line) {
    if (!f) {
        char s[10000];
        sprintf_s(s, "\n\n%s(%i) : Assertion failed : %s.\n\n",
            file, line, expr);
        puts(s);
        OutputDebugStringA(s);
        logger(s);
        return false;
    }
    return true;
}

#include "CUDADefines.h"

#define xyz(p) p.x, p.y, p.z
dim3 _lastLaunch_gridDim, _lastLaunch_blockDim;
/// \returns true if err is cudaSuccess
/// Fills errmsg in UNIT_TESTING build.
bool cudaSafeCallImpl(cudaError err, const char * const expr, const char * const file, const int line)
{
    if (cudaSuccess == err) return true;

    char s[10000];
    cudaGetLastError(); // Reset error flag
    const char* e = cudaGetErrorString(err);
    if (!e) e = "! cudaGetErrorString returned 0 !";

    sprintf_s(s, "\n%s(%i) : cudaSafeCall(%s)\nRuntime API error : %s.\n",
        file,
        line,
        expr,
        e);

    if (err == cudaError::cudaErrorInvalidConfiguration) {
        printf("configuration was (%d,%d,%d), (%d,%d,%d)\n",
            xyz(_lastLaunch_gridDim),
            xyz(_lastLaunch_blockDim)
            );
    }

    puts(s);
    OutputDebugStringA(s);
    logger(s);
    return false;
}

#if UNIT_TESTING

#else



/// Catch remaining cuda errors on shutdown
struct CATCHCUDAERRORS {
    ~CATCHCUDAERRORS() {
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());
        cudaSafeCall(cudaGetLastError());
    }
} _CATCHCUDAERRORS;


struct CATCHMALLOCERRORS {
    CATCHMALLOCERRORS() {
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
    }
} _CATCHMALLOCERRORS;
#endif

// TODO overload new/delete, new[], delete[] globally to add features:
// http://stackoverflow.com/a/1010811/524504
/*
sentry values (guards, magic)
alloc fill
free fill, catch double-free
delayed free
tracking (where are things allocated and destroyed, for whom)
*/
// note that dlls might use their own heap management 
// and talk directly to the os under the hood, 
// so you might not catch all

// also consider using Application Verifier
