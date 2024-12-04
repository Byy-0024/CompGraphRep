#ifndef GRAPH_CUH
#define GRAPH_CUH

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "../Graph.hpp"
#include "../Interval.hpp"

struct cuda_csr_graph{
    node *d_adjacencyList;
    size_t *d_edgesOffset;
    size_t *d_edgesSize;
	int num_nodes, num_edges;
    void init(const Graph &G);
    void free();
};

struct cuda_coo_graph{
    node *d_sourceNodesList;
    node *d_targetNodesList;
    size_t *d_edgesOffset;
    size_t *d_edgesSize;
    size_t num_nodes, num_edges;
    void init(const Graph &G);
    void free();
};

struct cuda_hvi_graph{
    HybridVertexInterval *d_HVIList;
    size_t *d_HVIOffset; 
    int num_nodes, num_hvi;
    void init(const HVI &G);
    void free();
};

struct cuda_hvi_coo_graph{
    node *d_sourceNodesList;
    HybridVertexInterval *d_HVIList;
    size_t *d_HVIOffset;
    size_t num_nodes, num_hvi;
    void init(const HVI &G);
    void free(); 
};

extern __device__ void cnt_intersection (node u, node v, node *targetNodesList, size_t *edgesOffset, size_t *edgesSize, ull *res);
extern __device__ void cnt_intersection_hvi (node u, node v, HybridVertexInterval *HVIList, size_t *HVIOffset, ull *res);
extern __device__ void has_edge(node u, node v, node *adjacencyList, size_t *edgesOffset, size_t *edgesSize, bool *res);
extern __device__ void has_edge (node u, node v, HybridVertexInterval *HVIList, size_t *HVIOffset, bool *res);
#endif // GRAPH_CUH