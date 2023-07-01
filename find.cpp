
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

#include <immintrin.h>

#include "utils.h"

template <class T, typename F>
void test_find_function_cold_cache(std::size_t max_elements, std::size_t trials,
                                   F find, std::size_t raw_size = 1UL << 27UL) {

  auto raw_elements = get_sorted<T>(raw_size);
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::mt19937_64 eng_seeded(0);

  std::map<size_t, std::map<size_t, size_t>> time_map;

  for (std::size_t element_count = 8; element_count < max_elements;
       element_count *= 2) {
    auto queires = get_sorted<T>(element_count);
    std::shuffle(queires.begin(), queires.end(), eng_seeded);
    std::uniform_int_distribution<T> dist(0, (T)(raw_size - element_count));
    for (std::size_t query_count = 1; query_count < 8; query_count *= 2) {
      size_t total_time = 0;
      for (std::size_t trial = 0; trial < trials; trial++) {
        T data_start = dist(eng);
        size_t start = get_time();
        size_t count_found = 0;
        for (size_t j = 0; j < query_count; j++) {
          T query = queires[j] + data_start;
          if (T(find(raw_elements.data() + data_start,
                     raw_elements.data() + data_start + element_count, query) -
                raw_elements.data()) == query) {
            count_found += 1;
          }
        }
        size_t end = get_time();
        if (count_found != query_count) {
          std::cerr << "didn't find all the elemnts, something wrong with the "
                       "find function\n";
        }
        total_time += end - start;
      }
      time_map[element_count][query_count] = total_time / trials;
    }
  }

  std::cout << "query count, ";
  for (auto &[query_count, time] : (*(time_map.rbegin())).second) {
    std::cout << query_count << ", ";
  }
  std::cout << "\n";

  for (auto &[element_count, inner_map] : time_map) {
    std::cout << element_count << ", ";
    for (auto &[query_count, time] : inner_map) {
      std::cout << ((double)time) / query_count << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <class T> T *linear_count_find(T *start, T *end, T &element) {
  for (auto it = start; it < end; it++) {
    if (*it == element) {
      return it;
    }
  }
  return end;
}

template <class T> T *vectorized_linear_find(T *start, T *end, T &element) {
  size_t count = 0;
  auto it = start;
#ifdef __AVX2__
  if constexpr (std::is_same_v<T, uint32_t>) {
    auto cmp = _mm256_set1_epi32(element);
    auto length = end - start;
    if (length > 8) {
      while (it < end - 8) {
        uint32_t block = _mm256_movemask_epi8(
            _mm256_cmpeq_epi32(cmp, _mm256_loadu_si256((__m256i *)it)));
        if (block) {
          return it + bsf_word(block) / 4;
        }
        it += 8;
      }
    }
  }

  if constexpr (std::is_same_v<T, uint64_t>) {
    auto cmp = _mm256_set1_epi64x(element);
    auto length = end - start;
    if (length > 4) {
      while (it < end - 4) {
        uint32_t block = _mm256_movemask_epi8(
            _mm256_cmpeq_epi64(cmp, _mm256_loadu_si256((__m256i *)it)));
        if (block) {
          return it + bsf_word(block) / 8;
        }
        it += 4;
      }
      count /= 8;
    }
  }

#endif
  for (; it < end; it++) {
    if (*it == element) {
      return it;
    }
  }
  return end;
}

template <class T>
T *vectorized_linear_find_manual_unroll(T *start, T *end, T &element) {
  size_t count = 0;
  auto it = start;
#ifdef __AVX2__
  if constexpr (std::is_same_v<T, uint32_t>) {
    auto cmp = _mm256_set1_epi32(element);
    auto length = end - start;
    if (length > 8) {
      if (length > 32) {
        auto blend_mask =
            _mm256_set_epi8(0x80, 0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80,
                            0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80, 0,
                            0x80, 0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80, 0);
        while (it < end - 32) {
          auto block1 =
              _mm256_cmpeq_epi32(cmp, _mm256_loadu_si256((__m256i *)it));
          auto block2 =
              _mm256_cmpeq_epi32(cmp, _mm256_loadu_si256((__m256i *)(it + 8)));
          auto block3 =
              _mm256_cmpeq_epi32(cmp, _mm256_loadu_si256((__m256i *)(it + 16)));
          auto block4 =
              _mm256_cmpeq_epi32(cmp, _mm256_loadu_si256((__m256i *)(it + 24)));
          auto blend1 = _mm256_blend_epi16(block1, block3, 0xAAU);
          auto blend2 = _mm256_blend_epi16(block2, block4, 0xAAU);
          auto blend = _mm256_movemask_epi8(
              _mm256_blendv_epi8(blend1, blend2, blend_mask));
          if (blend) {
            uint32_t bit_mask = bsf_word(blend);
            uint32_t word_index = bit_mask % 4;
            uint32_t bit_index = bit_mask / 4;
            return it + word_index * 8 + bit_index;
          }
          it += 32;
        }
      }
      while (it < end - 8) {
        uint32_t block = _mm256_movemask_epi8(
            _mm256_cmpeq_epi32(cmp, _mm256_loadu_si256((__m256i *)it)));
        if (block) {
          return it + bsf_word(block) / 4;
        }
        it += 8;
      }
    }
  }

  if constexpr (std::is_same_v<T, uint64_t>) {
    auto cmp = _mm256_set1_epi64x(element);
    auto length = end - start;
    if (length > 4) {
      while (it < end - 4) {
        uint32_t block = _mm256_movemask_epi8(
            _mm256_cmpeq_epi64(cmp, _mm256_loadu_si256((__m256i *)it)));
        if (block) {
          return it + bsf_word(block) / 8;
        }
        it += 4;
      }
      count /= 8;
    }
  }

#endif
  for (; it < end; it++) {
    if (*it == element) {
      return it;
    }
  }
  return end;
}

int main(int32_t argc, char *argv[]) {

  uint64_t max_elements = std::strtol(argv[1], nullptr, 10);

  std::cout << "std::find\n";
  {
    auto find = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return std::find(s, t, v);
    };
    std::cout << "uint32_t\n";
    test_find_function_cold_cache<uint32_t>(max_elements, 10, find);
    // std::cout << "uint64_t\n";
    // test_find_function_cold_cache<uint64_t>(max_elements, 10, find);
  }

  std::cout << "count_find\n";
  {
    auto find = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return linear_count_find(s, t, v);
    };
    std::cout << "uint32_t\n";
    test_find_function_cold_cache<uint32_t>(max_elements, 10, find);
    // std::cout << "uint64_t\n";
    // test_find_function_cold_cache<uint64_t>(max_elements, 10, find);
  }

  std::cout << "vectorized_linear_find\n";
  {
    auto find = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return vectorized_linear_find(s, t, v);
    };
    std::cout << "uint32_t\n";
    test_find_function_cold_cache<uint32_t>(max_elements, 10, find);
    // std::cout << "uint64_t\n";
    // test_find_function_cold_cache<uint64_t>(max_elements, 10, find);
  }

  std::cout << "vectorized_linear_find_manual_unroll\n";
  {
    auto find = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return vectorized_linear_find_manual_unroll(s, t, v);
    };
    std::cout << "uint32_t\n";
    test_find_function_cold_cache<uint32_t>(max_elements, 10, find);
    // std::cout << "uint64_t\n";
    // test_find_function_cold_cache<uint64_t>(max_elements, 10, find);
  }
}