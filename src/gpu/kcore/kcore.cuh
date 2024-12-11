#ifndef KCORE_CUH
#define KCORE_CUH

#include <bits/stdc++.h>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "../graph.cuh"

/*
 * An implementation of the ICDE 2023 paper "Accelerating k-Core Decomposition by a GPU".
 */

void kcoreGPU(cuda_hvi_graph G, std::vector<int> &core_numbers);
void kcoreGPU(cuda_csr_graph G, std::vector<int> &core_numbers);

#endif // KCORE_CUH