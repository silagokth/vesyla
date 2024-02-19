/**
 * This file contains the utility functions for debugging.
 */

#include "Array.hpp"
#include "Stream.hpp"

/**
 * Print the contents of a vector.
 * @param vec The array to print.
 */
template <typename T> void print_vector(std::vector<T> vec) {
  for (auto i : vec) {
    std::cout << i << ",";
  }
  std::cout << std::endl;
}

template <typename T> T decode_bitset_token(bitset<sizeof(T) * 8> raw_data_) {
  T token = 0;
  if (sizeof(T) == sizeof(uint64_t)) {
    uint64_t x = (*(uint64_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else if (sizeof(T) == sizeof(uint32_t)) {
    uint32_t x = (*(uint32_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else if (sizeof(T) == sizeof(uint16_t)) {
    uint16_t x = (*(uint16_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else if (sizeof(T) == sizeof(uint8_t)) {
    uint8_t x = (*(uint8_t *)(&token)) | raw_data_.to_ullong();
    token = *(T *)(&x);
  } else {
    abort();
  }
  return token;
}

template <size_t chunk_size>
vector<std::bitset<chunk_size>> chop_bitset(std::bitset<chunk_size> raw_data_,
                                            size_t size_) {
  assert(size_ <= chunk_size);
  assert(chunk_size % size_ == 0);
  vector<std::bitset<chunk_size>> vec;
  for (size_t i = 0; i < chunk_size / size_; i++) {
    std::bitset<chunk_size> section_i = 0;
    for (size_t j = 0; j < size_; j++) {
      section_i[j] = raw_data_[i * size_ + j];
    }
    vec.push_back(section_i);
  }
  return vec;
}

template <size_t chunk_size, typename T>
vector<T> decode_bitset(std::bitset<chunk_size> raw_data_) {
  vector<std::bitset<chunk_size>> vec =
      chop_bitset<chunk_size>(raw_data_, sizeof(T) * 8);
  vector<T> dec_vec(vec.size(), 0);
  for (size_t i = 0; i < vec.size(); i++) {
    std::bitset<sizeof(T) * 8> tt = vec[i].to_ulong();
    dec_vec[i] = decode_bitset_token<T>(tt);
  }
  return dec_vec;
}

/**
 * Print the contents of a stream of bitset.
 * @param stream The stream to print.
 */
template <size_t chunk_size, typename T>
void print_bitset_stream(Stream<bitset<chunk_size>> stream) {
  for (size_t i = 0; i < stream.size(); i++) {
    auto raw_data = stream[i];
    auto vec = decode_bitset<chunk_size, T>(raw_data);
    print_vector<T>(vec);
  }
}

// /**
//  * Print the contents of an array of bitset.
//  * @param arr The array to print.
//  */
// template <size_t chunk_num, size_t chunk_size, typename T>
// void print_bitset_array(Array<chunk_num, chunk_size> &arr, size_t addr_,
//                         size_t size_) {
//   std::vector<T> vec = arr.read<T>(addr_, size_);
//   print_vector<T>(vec);
// }