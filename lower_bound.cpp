
#include <algorithm>
#include <cstddef>
#include <functional>
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
  std::cout << "element size is " << sizeof(T) << "\n";

  auto raw_elements = get_sorted<T>(raw_size);
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::mt19937_64 eng_seeded(0);

  std::map<size_t, std::map<size_t, size_t>> time_map;

  for (std::size_t element_count = 8; element_count < max_elements;
       element_count *= 2) {
    std::uniform_int_distribution<size_t> dist(0, (raw_size - element_count));
    for (std::size_t query_count = 1; query_count < 8; query_count *= 2) {
      size_t total_time = 0;
      for (std::size_t trial = 0; trial < trials; trial++) {
        auto queires =
            create_random_data<size_t>(query_count, element_count, eng_seeded);
        auto data_start = dist(eng);
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

template <typename ForwardIterator, typename Tp>
_GLIBCXX20_CONSTEXPR ForwardIterator lb(ForwardIterator first,
                                        ForwardIterator last, const Tp &val) {
  typedef typename std::iterator_traits<ForwardIterator>::difference_type
      DistanceType;

  DistanceType len = std::distance(first, last);

  while (len > 4) {
    DistanceType quarter = len >> 2;
    ForwardIterator one_quarter = first;
    std::advance(one_quarter, quarter);
    ForwardIterator second_quarter = one_quarter;
    std::advance(second_quarter, quarter);
    ForwardIterator third_quarter = second_quarter;
    std::advance(third_quarter, quarter);
    bool first_part = !(*one_quarter < val);
    bool second_part = !(*second_quarter < val);
    bool third_part = !(*third_quarter < val);
    if (first_part) {
      len = std::distance(first, one_quarter);
    } else if (second_part) {
      first = one_quarter;
      len = std::distance(one_quarter, second_quarter);
    } else if (third_part) {
      first = second_quarter;
      len = std::distance(second_quarter, third_quarter);
    } else {
      first = third_quarter;
      len = len - std::distance(first, third_quarter);
    }
  }
  last = first;
  std::advance(last, len);
  while (first < last) {
    if (!(*first < val)) {
      return first;
    }
    ++first;
  }
  return first;
}

int main(int32_t argc, char *argv[]) {

  uint64_t max_elements = std::strtol(argv[1], nullptr, 10);

  std::cout << "std::lower_bound\n";
  {
    auto l_b = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return std::lower_bound(s, t, v);
    };
    std::cout << "uint32_t\n";
    test_find_function_cold_cache<uint32_t>(max_elements, 10, l_b);
    std::cout << "uint64_t\n";
    test_find_function_cold_cache<uint64_t>(max_elements, 10, l_b);
    std::cout << "16 bytes\n";
    test_find_function_cold_cache<wide_int<uint64_t, 16>>(max_elements, 10,
                                                          l_b);
    std::cout << "32 bytes\n";
    test_find_function_cold_cache<wide_int<uint64_t, 32>>(max_elements, 10,
                                                          l_b);
    std::cout << "64 bytes\n";
    test_find_function_cold_cache<wide_int<uint64_t, 64>>(max_elements, 10,
                                                          l_b);
  }

  std::cout << "lower_bound\n";
  {
    auto l_b = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return lb(s, t, v);
    };
    std::cout << "uint32_t\n";
    test_find_function_cold_cache<uint32_t>(max_elements, 10, l_b);
    std::cout << "uint64_t\n";
    test_find_function_cold_cache<uint64_t>(max_elements, 10, l_b);
    std::cout << "32 bytes\n";
    test_find_function_cold_cache<wide_int<uint64_t, 32>>(max_elements, 10,
                                                          l_b);
    std::cout << "64 bytes\n";
    test_find_function_cold_cache<wide_int<uint64_t, 64>>(max_elements, 10,
                                                          l_b);
  }
}