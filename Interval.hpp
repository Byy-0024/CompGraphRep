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

class HVI {
public:
    HVI(const Graph &g);
    bool bsearch(node u, node v, size_t pos) const;
    bool has_directed_edge(const node u, const node v) const;
    bool has_edge(node u, node v) const;
    void prefetch_neighbor(node u);
    void print_neighbor(node u) const;
    void bfs(node u, vector<int> &bfs_rank);
    void get_common_neighbor(node u, node v, vector<HybridVertexInterval> &res) const;
    void page_rank(vector<double> &res, double epsilon, size_t max_iter);
    size_t cnt_common_neighbor(node u, node v) const;
    size_t size_in_bytes() const;
    size_t cnt_tri_merge() const;
    size_t get_num_nodes() const;
    size_t get_num_hvi() const;
    size_t* get_inds() const;
    HybridVertexInterval* get_vals() const;
private:
    size_t num_nodes, num_hybrid_vertex_intervals;
    size_t* inds;
    HybridVertexInterval* vals;
};

bool hybrid_vid_greater(const HybridVertexInterval &x, const HybridVertexInterval &y);
bool hybrid_vid_less(const HybridVertexInterval &x, const HybridVertexInterval &y);
bool bfs_fast_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank);
bool bfs_fast_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank, vector<node> &origin_to_new);
void bfs_full_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank);
void bfs_full_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank, vector<node> &origin_to_new);

#endif // Interval_HPP