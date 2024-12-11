#ifndef CC_CUH
#define CC_CUH

#include <bits/stdc++.h>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "../graph.cuh"

/*
 * The Shiloach-Vishkin algorithm for connected components (hooking and shortcutting).
 * component_ids[i] = j means that vertex i is in component j.
 */
void ccShiloachVishkin(cuda_hvi_graph G, std::vector<int> &component_ids);
void ccShiloachVishkin(cuda_csr_graph G, std::vector<int> &component_ids);

#endif // CC_CUH