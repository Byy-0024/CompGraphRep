#include "cc.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5)

__global__
void hooking(cuda_csr_graph *g, int *d_component_ids, bool *d_hook) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < g->num_nodes) {
        for (int i = g->d_edgesOffset[tid]; i < g->d_edgesOffset[tid] + g->d_edgesSize[tid]; ++i) {
            int v = g->d_adjacencyList[i];
            if (d_component_ids[tid] < d_component_ids[v] && 
                d_component_ids[v] == d_component_ids[d_component_ids[v]]) {
                d_component_ids[d_component_ids[v]] = d_component_ids[tid];
                *d_hook = true;
            }
        }
	}
}

__global__
void shortcutting(cuda_csr_graph *g, int *d_component_ids) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < g->num_nodes) {
        while (d_component_ids[tid] != d_component_ids[d_component_ids[tid]])
            d_component_ids[tid] = d_component_ids[d_component_ids[tid]];
    }
}

__global__
void hooking(cuda_hvi_graph *g, int *d_component_ids, bool *d_hook) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < g->num_nodes) {
        for (int i = g->d_HVIOffset[tid]; i < g->d_HVIOffset[tid+1]; ++i) {
            HybridVertexInterval v = g->d_HVIList[i];
            if (v & HVI_LEFT_BOUNDARY_MASK) {
                int left = (v & GETNODE_MASK);
                v = g->d_HVIList[++i];
                int right = (v & GETNODE_MASK);
                for (int j = left; j <= right; ++j) {
                    if (d_component_ids[tid] < d_component_ids[j] && 
                        d_component_ids[j] == d_component_ids[d_component_ids[j]]) {
                        d_component_ids[d_component_ids[j]] = d_component_ids[tid];
                        *d_hook = true;
                    }
                }
            }
            else {
                if (d_component_ids[tid] < d_component_ids[v] &&
                    d_component_ids[v] == d_component_ids[d_component_ids[v]]) {
                    d_component_ids[d_component_ids[v]] = d_component_ids[tid];
                    *d_hook = true;
                }
            }
        }
    }
}

__global__
void shortcutting(cuda_hvi_graph *g, int *d_component_ids) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < g->num_nodes) {
        while (d_component_ids[tid] != d_component_ids[d_component_ids[tid]])
            d_component_ids[tid] = d_component_ids[d_component_ids[tid]];
    }
}

void ccShiloachVishkin(cuda_csr_graph g, vector<int> &component_ids) {
    bool hook = true;
    const int n = g.num_nodes;
    const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    component_ids.resize(n);
    for (int i = 0; i < n; ++i) component_ids[i] = i;      

    bool *d_hook;
    cudaMalloc(&d_hook, sizeof(bool));

    int *d_component_ids;
    cudaMalloc(&d_component_ids, n * sizeof(int));
    cudaMemcpy(d_component_ids, component_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cuda_csr_graph *d_g;
    cudaMalloc(&d_g, sizeof(cuda_csr_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_csr_graph), cudaMemcpyHostToDevice);

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(N_THREADS_PER_BLOCK);
    
    while (hook) {
        hook = false;
        cudaMemcpy(d_hook, &hook, sizeof(bool), cudaMemcpyHostToDevice);
        errorCheck(cudaDeviceSynchronize(), "initializing");
        hooking<<<dimGrid, dimBlock>>>(d_g, d_component_ids, d_hook);
        errorCheck(cudaDeviceSynchronize(), "hooking");
        shortcutting<<<dimGrid, dimBlock>>>(d_g, d_component_ids);
        errorCheck(cudaDeviceSynchronize(), "shortcutting");
        cudaMemcpy(&hook, d_hook, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(component_ids.data(), d_component_ids, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_g);
    cudaFree(d_component_ids);
    cudaFree(d_hook);
}

void ccShiloachVishkin(cuda_hvi_graph g, vector<int> &component_ids) {
    bool hook = true;
    const int n = g.num_nodes;
    const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    component_ids.resize(n);
    for (int i = 0; i < n; ++i) component_ids[i] = i;      

    bool *d_hook;
    cudaMalloc(&d_hook, sizeof(bool));

    int *d_component_ids;
    cudaMalloc(&d_component_ids, n * sizeof(int));
    cudaMemcpy(d_component_ids, component_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cuda_hvi_graph *d_g;
    cudaMalloc(&d_g, sizeof(cuda_hvi_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_hvi_graph), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(n_blocks);
    dim3 dimBlock(N_THREADS_PER_BLOCK);

    while (hook) {
        hook = false;
        cudaMemcpy(d_hook, &hook, sizeof(bool), cudaMemcpyHostToDevice);
        errorCheck(cudaDeviceSynchronize(), "initializing");
        hooking<<<dimGrid, dimBlock>>>(d_g, d_component_ids, d_hook);
        errorCheck(cudaDeviceSynchronize(), "hooking");
        shortcutting<<<dimGrid, dimBlock>>>(d_g, d_component_ids);
        errorCheck(cudaDeviceSynchronize(), "shortcutting");
        cudaMemcpy(&hook, d_hook, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(component_ids.data(), d_component_ids, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_component_ids);
    cudaFree(d_hook);
}