#include "bfs.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 7)

__global__
void compute_next_layer_distance(int n, node *adjacencyList, int *edgesOffset, int *edgesSize, int *distance, int level, bool *done) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (distance[tid] == level) {
			for (int i = edgesOffset[tid]; i < edgesOffset[tid] + edgesSize[tid]; ++i) {
				int v = adjacencyList[i];
				if (distance[v] == INT_MAX) {
					*done = false;
					distance[v] = level + 1;
				}
			}
		}
	}
}

// __global__
// void compute_next_layer_vertex(int n, cuda_compress_graph *d_g_comp, int *distance, int level, bool *done) {
// 	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (tid < n) {
// 		if (distance[tid] == level) {
// 			for (int i = d_g_comp->d_csr_vlist[tid]; i < d_g_comp->d_csr_vlist[tid+1]; ++i) {
// 				int v = d_g_comp->d_csr_elist[i];
// 				if (distance[v] == INT_MAX) {
// 					*done = false;
// 					distance[v] = level + 1;
// 				}
// 			}
// 		}
// 	}
// }

// __global__
// void compute_next_layer_rule(int k, int n, cuda_compress_graph *d_g_comp, int *distance, int level, bool *done) {
// 	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (tid < k) {
// 		int rule = tid + n;
// 		if (distance[rule] == level) {
// 			for (int i = d_g_comp->d_csr_vlist[rule]; i < d_g_comp->d_csr_vlist[rule+1]; ++i) {
// 				int v = d_g_comp->d_csr_elist[i];
// 				if (distance[v] == INT_MAX) {
// 					*done = false;
// 					distance[v] = level;
// 				}
// 			}
// 		}
// 	}
// }

__global__
void compute_next_layer_distance(int n, HybridVertexInterval *HybridVertexIntervalsList, int *HybridVertexIntervalsOffset, int *distance, int level, bool *done) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (distance[tid] == level) {
			node curr_head = 0;
			for (int i = HybridVertexIntervalsOffset[tid]; i < HybridVertexIntervalsOffset[tid+1]; ++i) {
				HybridVertexInterval v = HybridVertexIntervalsList[i];
				if (v & HVI_LEFT_BOUNDARY_MASK) {
                    curr_head = v ^ HVI_LEFT_BOUNDARY_MASK;
                    continue;
                }
                else if (v & HVI_RIGHT_BOUNDARY_MASK) {
                    node next = curr_head;
                    node right = v ^ HVI_RIGHT_BOUNDARY_MASK;
                    while (next <= right) {
                        if (distance[next] == INT_MAX) {
                            distance[next] = level + 1;
							*done = false;
                        }
                        next++;
                    }
                }
                else {
                    if (distance[v] == INT_MAX) {
                        distance[v] = level + 1;
						*done = false;
                    }
                }
			}
		}
	}
}

// Assumes that distance is a vector of all INT_MAX (except at start position)
void bfsGPUQuadratic(int start, cuda_csr_graph &G, vector<int> &distance) {
	const int n = G.num_nodes;
	// const int m = G.num_edges;
	const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

	// Initialization of GPU variables
	int *d_distance; // output
	bool done = false;
	const bool true_value = true;

	// Initialization of CPU variables
	bool *d_done;
	int level = 0;

	// Allocation on device
	cudaMalloc((void **)&d_distance, n * sizeof(int));
	cudaMalloc((void **)&d_done, sizeof(bool));  // malloc of single value is also important
	
	// auto startTime = chrono::steady_clock::now();
	distance = vector<int>(n, INT_MAX);
	distance[start] = 0;
	cudaMemcpy(d_distance, distance.data(), n * sizeof(int), cudaMemcpyHostToDevice);

	while (!done) {
		cudaMemcpy(d_done, &true_value, sizeof(bool), cudaMemcpyHostToDevice);
		compute_next_layer_distance <<<n_blocks, N_THREADS_PER_BLOCK>>> (n, G.d_adjacencyList, G.d_edgesOffset, G.d_edgesSize, d_distance, level, d_done);
		cudaDeviceSynchronize();
		cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
		++level;
		if (level > n) {
			cerr << "Number of iterations exceeded number of vertices!" << endl;
			break;
		}
	}

	// Copying output back to host
	cudaMemcpy(&distance[0], d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);
	// auto endTime = std::chrono::steady_clock::now();
	// auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	// printf("Elapsed time for quadratic GPU implementation (without copying graph) : %li ms.\n", duration);

	// Cleanup
	cudaFree(d_distance);
}

// void bfsGPUQuadratic(int start, cuda_compress_graph &G, vector<int> &distance) {
// 	const int n = G.num_nodes;
// 	const int k = G.num_rules;
// 	const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
// 	const int k_blocks = (k + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

// 	int *d_distance;
// 	bool done_vertex = false;
// 	bool done_rule = false;
// 	const bool true_value = true;

// 	bool *d_done_vertex, *d_done_rule;
// 	cuda_compress_graph *d_g;
// 	int level = 0;

// 	// Allocation on device
// 	cudaMalloc((void **)&d_distance, (n+k) * sizeof(int));
// 	cudaMalloc((void **)&d_done_vertex, sizeof(bool)); 
// 	cudaMalloc((void **)&d_done_rule, sizeof(bool));
// 	cudaMalloc((void **)&d_g, sizeof(cuda_compress_graph));
	
// 	distance = vector<int>(n+k, INT_MAX);
// 	distance[start] = 0;
// 	cudaMemcpy(d_distance, distance.data(), (n+k) * sizeof(int), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_g, &G, sizeof(cuda_compress_graph), cudaMemcpyHostToDevice);

// 	while (!done_vertex) {
// 		cudaMemcpy(d_done_vertex, &true_value, sizeof(bool), cudaMemcpyHostToDevice);
// 		compute_next_layer_vertex <<<n_blocks, N_THREADS_PER_BLOCK>>> (n, d_g, d_distance, level, d_done_vertex);
// 		cudaDeviceSynchronize();
// 		cudaMemcpy(&done_vertex, d_done_rule, sizeof(bool), cudaMemcpyDeviceToHost);
// 		done_rule = false;
// 		++level;
// 		while (!done_rule) {
// 			cudaMemcpy(d_done_rule, &true_value, sizeof(bool), cudaMemcpyHostToDevice);
// 			compute_next_layer_rule <<<k_blocks, N_THREADS_PER_BLOCK>>> (k, n, d_g, d_distance, level, d_done_rule);
// 			cudaDeviceSynchronize();
// 			cudaMemcpy(&done_rule, d_done_rule, sizeof(bool), cudaMemcpyDeviceToHost);
// 		}
// 		if (level > n) {
// 			cerr << "Number of iterations exceeded number of vertices!" << endl;
// 			break;
// 		}
// 	}

// 	// Copying output back to host
// 	cudaMemcpy(&distance[0], d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);
// 	// auto endTime = std::chrono::steady_clock::now();
// 	// auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
// 	// printf("Elapsed time for quadratic GPU implementation (without copying graph) : %li ms.\n", duration);

// 	// Cleanup
// 	cudaFree(d_distance);
// 	cudaFree(d_done_vertex);
// 	cudaFree(d_done_rule);
// 	cudaFree(d_g);
// }

void bfsGPUQuadratic(int start, cuda_hvi_graph &G, vector<int> &distance) {
	const int n = G.num_nodes;
	const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

	// Initialization of GPU variables
	int *d_distance; // output
	bool done = false;
	const bool true_value = true;

	// Initialization of CPU variables
	bool *d_done;
	int level = 0;

	// Allocation on device
	cudaMalloc((void **)&d_distance, n * sizeof(int));
	cudaMalloc((void **)&d_done, sizeof(bool));  // malloc of single value is also important
	
	// auto startTime = chrono::steady_clock::now();
	distance = vector<int>(n, INT_MAX);
	distance[start] = 0;
	cudaMemcpy(d_distance, distance.data(), n * sizeof(int), cudaMemcpyHostToDevice);

	while (!done) {
		cudaMemcpy(d_done, &true_value, sizeof(bool), cudaMemcpyHostToDevice);
		compute_next_layer_distance <<<n_blocks, N_THREADS_PER_BLOCK>>> (n, G.d_HVIList, G.d_HVIOffset, d_distance, level, d_done);
		cudaDeviceSynchronize();
		cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
		++level;
		if (level > n) {
			cerr << "Number of iterations exceeded number of vertices!" << endl;
			break;
		}
	}

	// Copying output back to host
	cudaMemcpy(&distance[0], d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);
	// auto endTime = std::chrono::steady_clock::now();
	// auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	// printf("Elapsed time for quadratic GPU implementation (without copying graph) : %li ms.\n", duration);

	// Cleanup
	cudaFree(d_distance);
}