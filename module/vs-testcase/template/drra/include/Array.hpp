// Copyright (C) 2022 Yu Yang
//
// This file is part of Vesyla.
//
// Vesyla is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Vesyla is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Vesyla.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __ARRAY_HPP__
#define __ARRAY_HPP__

#include <bitset>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

using namespace std;

template <size_t chunk_num, size_t chunk_size> class Array {
private:
  std::unordered_map<size_t, std::bitset<chunk_size>> _data;

public:
  Array() {}
  Array(std::unordered_map<size_t, std::bitset<chunk_size>> x_) { _data = x_; }

  void copy(Array<chunk_num, chunk_size> a_) { _data = a_.get_data(); }

  std::bitset<chunk_size> get_slice(size_t addr_) {
    assert(addr_ < chunk_num);
    if (_data.find(addr_) != _data.end()) {
      return _data[addr_];
    }
    return std::bitset<chunk_size>(0);
  }
  void set_slice(size_t addr_, std::bitset<chunk_size> slice_) {
    assert(addr_ < chunk_num);
    assert(slice_.size() == chunk_size);
    _data[addr_] = slice_;
  }
  vector<std::bitset<chunk_size>> chop(std::bitset<chunk_size> raw_data_,
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
  std::bitset<chunk_size> assemble(vector<std::bitset<chunk_size>> vec_,
                                   size_t size_) {
    assert(size_ <= chunk_size);
    assert(chunk_size % size_ == 0);
    std::bitset<chunk_size> raw_data;
    for (size_t i = 0; i < chunk_size / size_; i++) {
      for (size_t j = 0; j < size_; j++) {
        raw_data[i * size_ + j] = vec_[i][j];
      }
    }
    return raw_data;
  }

  template <typename T> T decode_token(bitset<sizeof(T) * 8> raw_data_) {
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

  template <typename T> vector<T> decode(std::bitset<chunk_size> raw_data_) {
    vector<std::bitset<chunk_size>> vec = chop(raw_data_, sizeof(T) * 8);
    vector<T> dec_vec(vec.size(), 0);
    for (size_t i = 0; i < vec.size(); i++) {
      std::bitset<sizeof(T) * 8> tt = vec[i].to_ulong();
      dec_vec[i] = decode_token<T>(tt);
    }
    return dec_vec;
  }
  string decode(std::bitset<chunk_size> raw_data_) {
    return raw_data_.to_string();
  }

  template <typename T> std::bitset<sizeof(T) * 8> encode_token(T token_) {
    std::bitset<sizeof(T) * 8> raw_data = 0;
    unsigned long long int n = 0;
    if (sizeof(T) == sizeof(uint64_t)) {
      n |= *(uint64_t *)(&token_);
    } else if (sizeof(T) == sizeof(uint32_t)) {
      n |= *(uint32_t *)(&token_);
    } else if (sizeof(T) == sizeof(uint16_t)) {
      n |= *(uint16_t *)(&token_);
    } else if (sizeof(T) == sizeof(uint8_t)) {
      n |= *(uint8_t *)(&token_);
    } else {
      abort();
    }
    raw_data = n;
    return raw_data;
  }
  template <typename T> std::bitset<chunk_size> encode(vector<T> enc_vec_) {
    vector<std::bitset<chunk_size>> vec(enc_vec_.size(), 0);
    for (size_t i = 0; i < vec.size(); i++) {
      vec[i] = encode_token<T>(enc_vec_[i]).to_ulong();
    }
    return assemble(vec, sizeof(T) * 8);
  }
  std::bitset<chunk_size> encode(string enc_str_) {
    std::bitset<chunk_size> raw_data(enc_str_);
    return raw_data;
  }

  template <typename T> void write(size_t addr_, size_t size_, vector<T> vec_) {
    assert(vec_.size() * sizeof(T) * 8 == size_ * chunk_size);
    assert(addr_ + size_ <= chunk_num);
    for (size_t i = 0; i < size_; i++) {
      vector<T> v(chunk_size / (sizeof(T) * 8));
      for (size_t j = 0; j < v.size(); j++) {
        v[j] = vec_[i * v.size() + j];
      }
      set_slice(addr_ + i, encode<T>(v));
    }
  }
  void write(size_t addr_, size_t size_, string enc_str_) {
    assert(enc_str_.size() == size_ * chunk_size);
    assert(addr_ + size_ <= chunk_num);
    for (size_t i = 0; i < size_; i++) {
      string v = enc_str_.substr(i * chunk_size, chunk_size);
      set_slice(addr_ + i, encode(v));
    }
  }

  template <typename T> vector<T> read(size_t addr_, size_t size_) {
    assert(addr_ + size_ <= chunk_num);
    vector<T> vec(size_ * chunk_size / (sizeof(T) * 8));
    for (size_t i = 0; i < size_; i++) {
      vector<T> v = decode<T>(get_slice(addr_ + i));
      for (size_t j = 0; j < v.size(); j++) {
        vec[i * v.size() + j] = v[j];
      }
    }
    return vec;
  }
  string read(size_t addr_, size_t size_) {
    assert(addr_ + size_ <= chunk_num);
    string str;
    for (size_t i = 0; i < size_; i++) {
      string v = decode(get_slice(addr_ + i));
      str += v;
    }
    return str;
  }

  size_t get_chunk_num() { return chunk_num; }
  size_t get_chunk_size() { return chunk_size; }
  size_t get_size() { return chunk_num * chunk_size; }
  size_t get_size_in_byte() { return chunk_num * chunk_size / 8; }
  std::unordered_map<size_t, std::bitset<chunk_size>> get_data() {
    return _data;
  }
  void reset() { _data.clear(); }
  bool is_active() {
    if (_data.size() > 0) {
      return true;
    }
    return false;
  }
  bool is_row_active(size_t addr_) {
    if (_data.find(addr_) != _data.end()) {
      return true;
    }
    return false;
  }
  vector<size_t> get_active_rows() {
    vector<size_t> active_rows;
    for (auto r : _data) {
      active_rows.push_back(r.first);
    }
    return active_rows;
  }
};

#endif // __ARRAY_HPP__
