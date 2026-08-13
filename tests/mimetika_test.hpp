#pragma once

#include <cstdio>
#include <utility>
#include <vector>

// Minimal zero-dependency unit-test harness: MIMETIKA_TEST cases
// self-register, MIMETIKA_TEST_MAIN() runs them all. One executable per
// tested header; ctest registers each executable as one test.

namespace mimetika_test {

inline int failures = 0;

using TestFn = void (*)();

inline std::vector<std::pair<const char*, TestFn>>& registry() {
  static std::vector<std::pair<const char*, TestFn>> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, TestFn fn) { registry().push_back({name, fn}); }
};

inline int run_all() {
  for (const auto& [name, fn] : registry()) {
    const int before = failures;
    fn();
    std::printf("[%s] %s\n", failures == before ? "PASS" : "FAIL", name);
  }
  if (failures != 0) std::printf("%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace mimetika_test

#define MIMETIKA_TEST(name)                                          \
  static void name();                                               \
  static ::mimetika_test::Registrar name##_registrar{#name, &name};  \
  static void name()

#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      ++::mimetika_test::failures;                                   \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                               \
  } while (0)

#define CHECK_THROWS(expr)                                          \
  do {                                                              \
    bool mimetika_test_threw = false;                                \
    try {                                                           \
      (void)(expr);                                                 \
    } catch (...) {                                                 \
      mimetika_test_threw = true;                                    \
    }                                                               \
    CHECK(mimetika_test_threw);                                      \
  } while (0)

#define MIMETIKA_TEST_MAIN() \
  int main() { return ::mimetika_test::run_all(); }
