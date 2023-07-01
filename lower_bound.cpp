
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
std::map<size_t, std::map<size_t, size_t>>
test_find_function_cold_cache(std::size_t max_elements, std::size_t trials,
                              F find, std::size_t raw_size = 1UL << 29UL) {
  std::cout << "element size is " << sizeof(T) << "\n";

  auto raw_elements = get_sorted<T>(raw_size);
  std::random_device rd;
  std::mt19937_64 eng(rd());

  std::map<size_t, std::map<size_t, size_t>> time_map;

  for (std::size_t element_count = 8; element_count < max_elements;
       element_count *= 2) {
    std::uniform_int_distribution<size_t> dist(0, (raw_size - element_count));
    for (std::size_t query_count = 1; query_count < element_count;
         query_count *= 16) {
      size_t total_time = 0;
      for (std::size_t trial = 0; trial < trials; trial++) {
        std::mt19937_64 eng_seeded(trial);
        auto queires =
            create_random_data<T>(query_count, element_count, eng_seeded);
        T data_start = dist(eng);
        size_t start = get_time();
        size_t count_found = 0;
        for (size_t j = 0; j < query_count; j++) {
          T query = static_cast<T>(queires[j] + data_start);
          if (static_cast<T>(
                  find(raw_elements.data() + data_start,
                       raw_elements.data() + data_start + element_count,
                       query) -
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
      std::cout << static_cast<double>(time) / static_cast<double>(query_count)
                << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  return time_map;
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
      len = quarter;
    } else if (second_part) {
      first = one_quarter;
      len = quarter;
    } else if (third_part) {
      first = second_quarter;
      len = quarter;
    } else {
      first = third_quarter;
      len = len - (quarter * 3);
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

template <class T, class... Ts>
void test_different_types(
    std::size_t max_elements, std::size_t trials, auto find,
    std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> &time_maps,
    std::size_t raw_size = 1UL << 30UL) {
  time_maps[sizeof(T)] =
      test_find_function_cold_cache<T>(max_elements, 10, find);
  if constexpr (sizeof...(Ts) > 0) {
    test_different_types<Ts...>(max_elements, trials, find, time_maps,
                                raw_size);
  }
}

static void compare_maps(
    std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> &time_maps1,
    std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> &time_maps2) {

  for (auto &[size, time_map1] : time_maps1) {
    std::cout << "element size = " << size << "\n";
    auto time_map2 = time_maps2[size];
    std::cout << "query count, ";
    for (auto &[query_count, time] : (*(time_map1.rbegin())).second) {
      std::cout << query_count << ", ";
    }
    std::cout << "\n";
    for (auto &[element_count, inner_map1] : time_map1) {
      auto inner_map2 = time_map2[element_count];
      std::cout << element_count << ", ";
      for (auto &[query_count, time1] : inner_map1) {
        auto time2 = inner_map2[query_count];
        std::cout << static_cast<double>(time1) / static_cast<double>(time2)
                  << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main(int32_t argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "tales in the single argument of max size\n";
    return -1;
  }

  uint64_t max_elements = std::strtol(argv[1], nullptr, 10);

  std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> time_maps_std;
  std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> time_maps_new;

  std::cout << "std::lower_bound\n";
  {
    auto l_b = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return std::lower_bound(s, t, v);
    };
    test_different_types<int32_t, int64_t, wide_int<int64_t, 16>,
                         wide_int<int64_t, 32>>(max_elements, 10, l_b,
                                                time_maps_std);
  }

  std::cout << "lower_bound\n";
  {
    auto l_b = [](auto s, auto t, auto v) __attribute__((noinline)) {
      return lb(s, t, v);
    };
    test_different_types<int32_t, int64_t, wide_int<int64_t, 16>,
                         wide_int<int64_t, 32>>(max_elements, 10, l_b,
                                                time_maps_new);
  }
  compare_maps(time_maps_std, time_maps_new);
}
