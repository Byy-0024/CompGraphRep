#ifndef GRAPH_HPP
#define GRAPH_HPP
#pragma once
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdio.h>
#include <string>
#include <string.h>
#include <unordered_set>
#include <vector>

typedef int node;
typedef unsigned long long ull;

struct time_counter {
    std::chrono::_V2::steady_clock::time_point t_start, t_end;
    ull t_cnt = 0;
    void start();
    void stop();
    void print(std::string s);
    void clean();
};

class Graph {
    public:
        Graph(const std::string graphfile);
        Graph(const std::string csr_vlist_binfile, const std::string csr_elist_binfile);
        Graph(const std::vector<std::vector<node>> &adj_list);
        
        // Operators.
        const int* get_csr_vlist() const;
        const node* get_csr_elist() const;
        bool has_edge(node u, node v) const;
        bool has_directed_edge(node u, node v) const;
        node get_nei(node u, size_t offset) const;  
        size_t cnt_common_neis(node u, node v) const;
        size_t cnt_common_neis(node u, const std::vector<node> &nodes) const;
        size_t get_num_edges() const;
        size_t get_num_nodes() const;
        size_t get_deg(node u) const;
        ull size_in_bytes() const;
        void get_common_neis(node u, node v, std::vector<node>& res) const;
	    void get_common_neis(node u, const std::vector<node> &nodes, std::vector<node> &res) const;
        void get_degs(std::vector<int> &degs) const;
        void get_neis(node u, std::vector<node> &neis) const;
        void print_neis(node u) const;
        void reorder(const std::vector<node> &origin2new);
        void save_as_csr_binfile(const std::string csr_vlist_binfile, const std::string csr_elist_binfile);

        // Algorithms.
        size_t tc();
        ull mc(node u);
        ull subgraph_matching(const Graph &q);
        void BKP(std::vector<node> &P, std::vector<node> &X, ull &res);
        void bfs(node u, std::vector<size_t> &bfs_rank);
        void cc(std::vector<node> &component_id);
        void core_decomposition(std::vector<size_t> &core_numbers);
        void dfs_helper(node u, std::vector<bool> &visited, std::vector<int> &new2origin) const;
        void get_dfs_order(node u, std::vector<int> &new2origin) const;
        void page_rank(std::vector<double> &res, double epsilon, size_t max_iter);
        void subgraph_matching_helper(const Graph &q, std::vector<node> &partial_matching, std::vector<int> &join_order, ull &res);
        void subgraph_matching_helper(const Graph &q, std::unordered_set<node> &partial_matching, std::vector<int> &join_order, std::vector<std::vector<node>> &inter_res, ull &res);

    private:
        size_t num_nodes, num_edges;
        std::vector<int> csr_vlist;
        std::vector<node> csr_elist;
};

template <class T>
bool weight_less(const std::pair<node, T> &x, const std::pair<node, T> &y);
template <class T>
bool weight_greater(const std::pair<node, T> &x, const std::pair<node, T> &y);
template <class T>
bool vid_less(const std::pair<node, T> &x, const std::pair<node, T> &y);
template <class T>
bool vid_greater(const std::pair<node, T> &x, const std::pair<node, T> &y);

#endif