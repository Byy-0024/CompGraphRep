#ifndef UTIL_HPP
#define UTIL_HPP
#pragma once
#include<algorithm>
#include<boost/dynamic_bitset.hpp>
#include<chrono>
#include<fstream>
#include"immintrin.h"
#include<iostream>
#include<map>
#include<numeric>
#include<omp.h>
#include<queue>
#include<sstream>
#include<stdio.h>
#include<string>
#include<string.h>
// #include<unordered_map>
#include<set>
#include<vector>

typedef size_t node;
typedef size_t weight;
typedef unsigned long long ull;

struct time_counter{
    void start();
    void stop();
    void print(std::string s);
    void clear();
    size_t _cnt = 0;
    std::chrono::steady_clock::time_point _t_start;
};

bool weight_less(const std::pair<node, size_t> &x, const std::pair<node, size_t> &y);
bool weight_greater(const std::pair<node, size_t> &x, const std::pair<node, size_t> &y);
bool vid_less(const std::pair<node, size_t> &x, const std::pair<node, size_t> &y);
bool vid_greater(const std::pair<node, size_t> &x, const std::pair<node, size_t> &y);

#endif // UTIL_HPP