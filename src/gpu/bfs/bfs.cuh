#ifndef BFS_CUH
#define BFS_CUH

#include <bits/stdc++.h>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "../graph.cuh"

/*
 * start - vertex number from which traversing a graph starts
 * G - graph to traverse
 * distance - placeholder for vector of distances (filled after invoking a function)
 * visited - placeholder for vector indicating that vertex was visited (filled after invoking a function)
 */
void bfsGPUQuadratic(int start, cuda_hvi_graph &G, std::vector<int> &distance);
void bfsGPUQuadratic(int start, cuda_compress_graph &G, std::vector<int> &distance);
void bfsGPUQuadratic(int start, cuda_csr_graph &G, std::vector<int> &distance);

#endif // BFS_CUH