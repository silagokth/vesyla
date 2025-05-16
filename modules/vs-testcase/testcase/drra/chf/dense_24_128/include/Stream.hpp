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

#ifndef __STREAM_HPP__
#define __STREAM_HPP__

#include <bitset>
#include <vector>

template<class T>
class Stream{
public:
    std::vector<T> data;

public:
    Stream(){}
    Stream(size_t size){
        data.resize(size);
    }
    size_t size(){
        return data.size();
    }
    T& operator[](size_t index)
    {
        return data[index];
    }
    void push_back(T e){
        data.push_back(e);
    }
};

#endif // __STREAM_HPP__