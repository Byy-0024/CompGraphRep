#ifndef TC_CUH
#define TC_CUH

#include <bits/stdc++.h>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "../graph.cuh"

void tcGPU(cuda_hvi_coo_graph G, ull &res);
void tcGPU(cuda_coo_graph G, ull &res);

#endif // TC_CUH