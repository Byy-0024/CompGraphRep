#ifndef PR_CUH
#define PR_CUH

#include <bits/stdc++.h>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "../graph.cuh"

void pageRank(cuda_hvi_graph &G, std::vector<float> &pr_val, int max_iter, float epsilon);
void pageRank(cuda_csr_graph &G, std::vector< float> &pr_val, int max_iter, float epsilon);

#endif // PR_CUH