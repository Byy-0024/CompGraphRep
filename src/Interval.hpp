#ifndef INTERVAL_HPP
#define INTERVAL_HPP
#pragma once
# include "Graph.hpp"

#define HVI_LEFT_BOUNDARY_MASK 0x80000000
#define HVI_RIGHT_BOUNDARY_MASK 0x40000000
#define GETNODE_MASK 0x3fffffff

// A HybridVertexInterval is a 32-bit unsigned integer, where the highest 2 bits mark its class (00: vertex, 10: left boundary, 11: right boundary) 
// and the rest of the 30 bits encodes the ID value. 
typedef uint32_t HybridVertexInterval;

// A Interval is a 64-bit unsigned integer, where the high 32 bits encode the left boundary and the low 32 bits encode the right boundary.
typedef uint64_t Interval;

class UI{
public:
     UI(const Graph &g);
    size_t size_in_bytes() const;
private:
    size_t* inds;
    Interval* intervals;
    size_t num_nodes, num_edges;
};

struct hvi_vector{
    node curr_back;
    std::vector<HybridVertexInterval> hvi_list;
    hvi_vector(const std::vector<HybridVertexInterval> &hvis);
    hvi_vector() : curr_back(0), hvi_list(std::vector<HybridVertexInterval>()) {};
    bool empty() const;
    bool has_element(node u) const;
    node back() const;
    size_t get_decoded_size() const;
    size_t get_decoded_size(std::vector<node> &nodes) const;
    size_t get_decoded_size(std::unordered_set<node> &nodes) const;
    size_t size() const;
    void clear();
    // void erase(node u);
    void pop();
    void print();
};

class HVI {
public:
    HVI(const Graph &g);
    HVI(const std::string hvi_offsets_binfile, const std::string hvi_list_binfile);

    // Operations
    bool bsearch(node u, node v, size_t pos) const;
    bool has_directed_edge(const node u, const node v) const;
    bool has_edge(node u, node v) const;
    HybridVertexInterval* get_hvi_list() const;
    HybridVertexInterval* get_neis(node u) const;
    size_t cnt_common_neighbor(node u, node v) const;
    size_t cnt_common_neighbor(node u, hvi_vector &hvis) const;
    size_t size_in_bytes() const;
    size_t get_num_nodes() const;
    size_t get_num_hvi() const;
    size_t get_degree(node u) const;
    size_t* get_hvi_offsets() const;
    void get_common_neis(node u, node v, std::vector<HybridVertexInterval> &res) const;
    void get_common_neis(node u, const hvi_vector &hvis, hvi_vector &res) const;
    void get_common_neis(node u, const std::vector<node> &nodes, std::vector<node> &res) const;
    void get_neis(node u, std::vector<node> &neis) const;
    void get_neis(node u, std::vector<HybridVertexInterval> &neis) const;
    void print_neighbor(node u) const;
    void save_as_binfile(const std::string hvi_offset_binfile, const std::string hvi_list_binfile) const;

    // Algorithms
    size_t tc() const;
    ull mc(node u);
    ull subgraph_matching(const Graph &q);
    void BKP(std::vector<node> &P, std::vector<node> &X, ull &res);
    void bfs(node u, std::vector<size_t> &bfs_rank);
    void cc(std::vector<int> &in_components);
    void core_decomposition(std::vector<size_t> &core_number);
    void page_rank(std::vector<double> &res, double epsilon, size_t max_iter);
    void subgraph_matching_helper(const Graph &q, std::vector<node> &partial_matching, std::vector<int> &join_order, const std::vector<size_t> &degs, ull &res);
    void subgraph_matching_helper(const Graph &q, std::unordered_set<node> &partial_matching, std::vector<int> &join_order, std::vector<hvi_vector> &inter_res, ull &res);
    
private:
    size_t num_nodes, num_hvis;
    size_t* hvi_offsets;
    HybridVertexInterval* hvi_list;
};
    
// struct hvi_iterator{
//     HybridVertexInterval *curr_pos;
//     HybridVertexInterval *end_pos;
//     node curr_val;
//     node right_val;
//     node get_val() const;
//     hvi_iterator operator++();
//     hvi_iterator(const HVI &g_hvi, node u);
// };

struct hvi_iterator{
    HybridVertexInterval *curr_pos;
    HybridVertexInterval *end_pos;
    node curr_val;
    // node step;
    node right_val;
    node get_val() const;
    hvi_iterator operator++();
    hvi_iterator(const HVI &g_hvi, node u);
};

bool hybrid_vid_greater(const HybridVertexInterval &x, const HybridVertexInterval &y);
bool hybrid_vid_less(const HybridVertexInterval &x, const HybridVertexInterval &y);
#endif // Interval_HPP