#include "ee.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 7)

__global__
void ee_kernel(const int n, cuda_csr_graph *d_g, int *queries, int *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sdata[N_THREADS_PER_BLOCK];  
    int t_res = 0;
    if (tid < n) {
        int u = queries[2 * tid], v = queries[2 * tid + 1];
        t_res = has_edge(u, v, d_g->d_adjacencyList, d_g->d_edgesOffset, d_g->d_edgesSize);
    }

    int local_id = threadIdx.x;
    sdata[local_id] = t_res;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            sdata[local_id] += sdata[local_id + s];
        }
        __syncthreads();
    }

    if (local_id == 0) {
        atomicAdd(res, sdata[0]);
    }
}

__global__
void ee_kernel(const int n, cuda_hvi_graph *d_g, int *queries, int *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sdata[N_THREADS_PER_BLOCK];  
    int t_res = 0;
    if (tid < n) {
        int u = queries[2 * tid], v = queries[2 * tid + 1];
        t_res = has_edge(u, v, d_g->d_HVIList, d_g->d_HVIOffset);
    }

    int local_id = threadIdx.x;
    sdata[local_id] = t_res;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            sdata[local_id] += sdata[local_id + s];
        }
        __syncthreads();
    }
    
    if (local_id == 0) {
        atomicAdd(res, sdata[0]);
    }
}

void eeGPU(cuda_csr_graph &g, const vector<node> &queries, int &res) {
    const int n = queries.size() / 2;
    const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    cuda_csr_graph *d_g;
    cudaMalloc(&d_g, sizeof(cuda_csr_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_csr_graph), cudaMemcpyHostToDevice);
    int *d_queries, *d_res;
    cudaMalloc(&d_queries, queries.size() * sizeof(int));
    cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_res, sizeof(int));
    cudaMemset(d_res, 0, sizeof(int));

    ee_kernel<<<n_blocks, N_THREADS_PER_BLOCK>>>(n, d_g, d_queries, d_res);
    errorCheck(cudaDeviceSynchronize(), "testing edge existance");
    cudaDeviceSynchronize();

    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_g);
    cudaFree(d_queries);
    cudaFree(d_res);
}

void eeGPU(cuda_hvi_graph &g, const vector<node> &queries, int &res) {
    const int n = queries.size() / 2;
    const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    cuda_hvi_graph *d_g;
    cudaMalloc(&d_g, sizeof(cuda_hvi_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_hvi_graph), cudaMemcpyHostToDevice);
    int *d_queries, *d_res;
    cudaMalloc(&d_queries, queries.size() * sizeof(int));
    cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_res, sizeof(int));
    cudaMemset(d_res, 0, sizeof(int));

    ee_kernel<<<n_blocks, N_THREADS_PER_BLOCK>>>(n, d_g, d_queries, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_g);
    cudaFree(d_queries);
    cudaFree(d_res);
}
