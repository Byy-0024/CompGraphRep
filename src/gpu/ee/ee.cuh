#ifndef EE_CUH
#define EE_CUH

#include <bits/stdc++.h>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "../graph.cuh"

/*
Edge existance test.
*/
void eeGPU(cuda_hvi_graph &G, const std::vector<node> &queries, int &res);
void eeGPU(cuda_csr_graph &G, const std::vector<node> &queries, int &res);

#endif // EE_CUH