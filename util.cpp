#include "util.hpp"

using namespace std;

void time_counter::start() {
    _t_start = chrono::steady_clock::now();
}

void time_counter::stop() {
    auto _t_end = chrono::steady_clock::now();
    _cnt += chrono::duration_cast<chrono::nanoseconds>(_t_end - _t_start).count();
}

void time_counter::print(string s) {
	if (_cnt > 10000000000) printf("Time used for %s: %.3f s.\n", s.c_str(), (float)_cnt / 1000000000);
	else if (_cnt > 10000000) printf("Time used for %s: %.3f ms.\n", s.c_str(), (float)_cnt / 1000000); 
	else if (_cnt > 10000) printf("Time used for %s: %.3f us.\n", s.c_str(), (float)_cnt / 1000);
	else printf("Time used for %s: %lu ns.\n", s.c_str(), _cnt);
}

void time_counter::clear() {
	_cnt = 0;
}

bool weight_less(const pair<node, size_t> &x, const pair<node, size_t> &y) {
    return x.second < y.second;
}

bool weight_greater(const pair<node, size_t> &x, const pair<node, size_t> &y) {
    return x.second > y.second;
}

bool vid_less(const pair<node, size_t> &x, const pair<node, size_t> &y) {
    return x.first < y.first;
}

bool vid_greater(const pair<node, size_t> &x, const pair<node, size_t> &y) {
    return x.first > y.first;
}