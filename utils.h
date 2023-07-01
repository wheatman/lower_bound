#pragma once

#include <limits>
#include <random>
#include <sys/time.h>
#include <vector>

template <class T> std::vector<T> get_sorted(std::size_t n) {
  std::vector<T> ret(n);
  for (std::size_t i = 0; i < n; i++) {
    ret[i] = i;
  }
  return ret;
}

template <class T>
std::vector<T> create_random_data(size_t n, size_t max_val,
                                  std::mt19937_64 eng) {
  std::uniform_int_distribution<uint64_t> dist(0,
                                               static_cast<uint64_t>(max_val));
  std::vector<T> v(n);
  for (auto &el : v) {
    el = dist(eng);
  }
  return v;
}

static inline uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
}

#if CYCLE_TIMER == 1
static inline uint64_t get_time() { return __rdtsc(); }
#else
static inline uint64_t get_time() { return get_usecs(); }
#endif

static inline uint32_t bsf_word(uint32_t word) {
  uint32_t result;
  __asm__("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

#include <immintrin.h>

template <class T> inline void Log(const __m256i &value) {
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256(static_cast<__m256i *>(buffer), value);
  for (size_t i = 0; i < n; i++) {
    std::cout << +buffer[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T, size_t alignment> class wide_int {
  alignas(alignment) T value;

public:
  wide_int(T v) : value(v) {}
  wide_int() : value(0) {}
  operator T() { return value; }

  auto operator<=>(const wide_int &b) const { return value <=> b.value; }
};
