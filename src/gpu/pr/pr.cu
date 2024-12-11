#include "pr.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 7)

__global__
void compute_an_iteration(cuda_csr_graph *d_g, float *pr_val, float *prev_pr_val) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < d_g->num_nodes) {
		pr_val[tid] = 0;
        for (int i = d_g->d_edgesOffset[tid]; i < d_g->d_edgesOffset[tid] + d_g->d_edgesSize[tid]; ++i) {
            node v = d_g->d_adjacencyList[i];
            pr_val[tid] += prev_pr_val[v]; 
        }
    }
}

__global__
void get_deg(cuda_hvi_graph *d_g, int *deg) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < d_g->num_nodes) {
		int res = 0;
		for (int i = d_g->d_HVIOffset[tid]; i < d_g->d_HVIOffset[tid+1];) {
            HybridVertexInterval v = d_g->d_HVIList[i];
            if (v & HVI_LEFT_BOUNDARY_MASK) {
                node left = v & GETNODE_MASK;
                node right = d_g->d_HVIList[i+1] & GETNODE_MASK;
                res += (right - left + 1);
                i += 2;
            }
            else {
                res++;
                i++;
            }
        }
		deg[tid] = res;
	}
}

__global__
void scale_by_deg(int n, float *pr_val, int *deg) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		pr_val[tid] /= (float) deg[tid];
	}
}

__global__
void compute_an_iteration(cuda_hvi_graph *d_g, float *pr_val, float *prev_pr_val) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < d_g->num_nodes) {
		pr_val[tid] = 0;
        for (int i = d_g->d_HVIOffset[tid]; i < d_g->d_HVIOffset[tid+1];) {
            HybridVertexInterval v = d_g->d_HVIList[i];
            if (v & HVI_LEFT_BOUNDARY_MASK) {
                node left = v & GETNODE_MASK;
                node right = d_g->d_HVIList[i+1] & GETNODE_MASK;
                for (node next = left; next <= right; next++) {
                    pr_val[tid] += prev_pr_val[next];
                }
                i += 2;
            }
            else {
                pr_val[tid] += prev_pr_val[v];
                i++;
            }
        }
	}
}

__global__
void compute_difference(const int n, float *pr_val, float *prev_pr_val, int *d_deg, float *d_diff) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sdata[N_THREADS_PER_BLOCK];
	float t_res = 0;
	if (tid < n) {
		if (d_deg[tid] > 0)	t_res = fabs(pr_val[tid] - prev_pr_val[tid] * d_deg[tid]);
	}

	const int local_idx = threadIdx.x;
	sdata[local_idx] = t_res;
	__syncthreads();

	for (unsigned int s = N_THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
		if (local_idx < s) {
			sdata[local_idx] += sdata[local_idx + s];
		}
		__syncthreads();
	}

	if (local_idx == 0) {
		atomicAdd(d_diff, sdata[0]);
	}
}

void pageRank(cuda_csr_graph &G, vector<float> &pr_val, int max_iter, float epsilon) {
	const int n = G.num_nodes;
	const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    cuda_csr_graph *d_g;
	float *d_pr_val; 
    float *d_prev_pr_val;
	float *d_diff;
	int iter = 0;
	float diff = (float) 1;

    cudaMalloc((void**)&d_g, sizeof(cuda_csr_graph));
	cudaMalloc((void **)&d_pr_val, n * sizeof(float));
	cudaMalloc((void **)&d_prev_pr_val, n * sizeof(float));
	cudaMalloc((void **)&d_diff, sizeof(float));
	
    cudaMemcpy(d_g, &G, sizeof(cuda_csr_graph), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pr_val, pr_val.data(), n * sizeof(float), cudaMemcpyHostToDevice);

	while (iter < max_iter) {
	// while (iter < max_iter && diff > epsilon) {
        cudaMemcpy(d_prev_pr_val, d_pr_val, n * sizeof(float), cudaMemcpyDeviceToDevice);
		scale_by_deg <<<n_blocks, N_THREADS_PER_BLOCK>>> (n, d_prev_pr_val, G.d_edgesSize);
		errorCheck(cudaDeviceSynchronize(), "scaling by degree");
		compute_an_iteration <<<n_blocks, N_THREADS_PER_BLOCK>>> (d_g, d_pr_val, d_prev_pr_val);
		errorCheck(cudaDeviceSynchronize(), "updating pr_val");
		cudaMemset(&d_diff, 0, sizeof(float));
		errorCheck(cudaDeviceSynchronize(), "initializing difference");
		compute_difference<<<n_blocks, N_THREADS_PER_BLOCK>>> (n, d_pr_val, d_prev_pr_val, G.d_edgesSize, d_diff);
		errorCheck(cudaDeviceSynchronize(), "computing difference");
		cudaMemcpy(&diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
		// printf("iter %d, diff %.3f\n", iter, diff);
		++iter;
	}

	cudaMemcpy(pr_val.data(), d_pr_val, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_g);
	cudaFree(d_pr_val);
    cudaFree(d_prev_pr_val);
	cudaFree(&d_diff);
}

void pageRank(cuda_hvi_graph &G, vector<float> &pr_val, int max_iter, float epsilon) {
	const int n = G.num_nodes;
	const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

	cuda_hvi_graph *d_g;
    float *d_pr_val; 
    float *d_prev_pr_val;
	int *d_deg;
	float *d_diff;
	int iter = 0;
	float diff = 1;

	cudaMalloc((void**)&d_g, sizeof(cuda_hvi_graph));
    cudaMalloc((void **)&d_pr_val, n * sizeof(float));
	cudaMalloc((void **)&d_prev_pr_val, n * sizeof(float));
	cudaMalloc((void **)&d_deg, n * sizeof(int));
	cudaMalloc((void **)&d_diff, sizeof(float));
    cudaMemcpy(d_g, &G, sizeof(cuda_hvi_graph), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pr_val, pr_val.data(), n * sizeof(float), cudaMemcpyHostToDevice);

	get_deg <<<n_blocks, N_THREADS_PER_BLOCK>>> (d_g, d_deg);

	while (iter < max_iter && diff > epsilon) {
        cudaMemcpy(d_prev_pr_val, d_pr_val, n * sizeof(float), cudaMemcpyDeviceToDevice);
		scale_by_deg <<<n_blocks, N_THREADS_PER_BLOCK>>> (n, d_prev_pr_val, d_deg);
		errorCheck(cudaDeviceSynchronize(), "scaling by degree");
		compute_an_iteration <<<n_blocks, N_THREADS_PER_BLOCK>>> (d_g, d_pr_val, d_prev_pr_val);
		errorCheck(cudaDeviceSynchronize(), "updating pr_val");
		cudaMemset(&d_diff, 0, sizeof(float));
		compute_difference<<<n_blocks, N_THREADS_PER_BLOCK>>> (n, d_pr_val, d_prev_pr_val, d_deg, d_diff);
		errorCheck(cudaDeviceSynchronize(), "computing difference");
		cudaMemcpy(&diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
		++iter;
	}

	// Copying output back to host
	cudaMemcpy(pr_val.data(), d_pr_val, n * sizeof(float), cudaMemcpyDeviceToHost);

	// Cleanup
	cudaFree(d_g);
	cudaFree(&d_diff);
	cudaFree(d_deg);
	cudaFree(d_pr_val);
    cudaFree(d_prev_pr_val);
}