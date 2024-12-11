#include "kcore.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 7)

__global__
void find_min_if_not_less(const int n, int *d_k, int *d_degs, int *d_min) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sdata[N_THREADS_PER_BLOCK];

    int t_res = n;
    if (tid < n) {
        if (d_degs[tid] >= d_k[0]) t_res = d_degs[tid];
    }

    const int local_id = threadIdx.x;
    sdata[local_id] = t_res;
    __syncthreads();

    for (int s = N_THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            sdata[local_id] = min(sdata[local_id + s], sdata[local_id]);
        }
        __syncthreads();
    }

    if (local_id == 0) {
        // printf("n: %d, tid: %d, sdata[0]: %d, min: %d\n", n, tid, sdata[0], d_min[0]);
        atomicMin(d_min, sdata[0]);
    }
}

// __global__
// void scan(cuda_csr_graph *d_g, int *d_degs, int *d_k, int *d_active) {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < d_g->num_nodes) {
//         if (d_degs[tid] == *d_k) {
//             d_active[tid] = 1;
//         }
//     }
// }

// __global__
// void allocate_buffer(const int n, int *d_buffer, int *d_buffer_sizes) {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < n) {
//         if (d_buffer_sizes[tid] < d_buffer_sizes[tid+1]) {
//             d_buffer[d_buffer_sizes[tid]] = tid;
//         }
//     }
// }

__global__
void scan(cuda_csr_graph *d_g, int *d_degs, int *d_buffer, int *d_buffer_size, int *d_k) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_g->num_nodes) {
        if (d_degs[tid] == *d_k) {
            int pos = atomicAdd(d_buffer_size, 1);
            d_buffer[pos] = tid;
        }
    }
}

__global__
void loop(cuda_csr_graph *d_g, int *d_degs, int *d_k, int *d_buffer, int *d_buffer_size, int *d_new_buffer, int *d_new_buffer_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *d_buffer_size) {
        int u = d_buffer[tid];
        for (int i = d_g->d_edgesOffset[u]; i < d_g->d_edgesOffset[u+1]; i++) {
            int v = d_g->d_adjacencyList[i];
            if (d_degs[v] > *d_k) {
                int deg_v = atomicSub(d_degs + v, 1);
                if (deg_v == *d_k + 1) {
                    int pos = atomicAdd(d_new_buffer_size, 1);
                    d_new_buffer[pos] = v;
                }
                if (deg_v <= *d_k) atomicAdd(d_degs + v, 1);
            }
        }
    }
}

__global__
void get_degrees(cuda_hvi_graph *d_g, int *d_degs) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_g->num_nodes) {
        for (int i = d_g->d_HVIOffset[tid]; i < d_g->d_HVIOffset[tid+1]; i++) {
            HybridVertexInterval hvi = d_g->d_HVIList[i];
            if (hvi & HVI_LEFT_BOUNDARY_MASK) {
                i++;
                d_degs[tid] += (d_g->d_HVIList[i] & GETNODE_MASK) - (hvi & GETNODE_MASK) + 1;
            }
            else {
                d_degs[tid]++;
            }
        }
    }
}

__global__
void scan(cuda_hvi_graph *d_g, int *d_degs, int *d_buffer, int *d_buffer_size, int *d_k) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_g->num_nodes) {
        if (d_degs[tid] == *d_k) {
            int pos = atomicAdd(d_buffer_size, 1);
            d_buffer[pos] = tid;
        }
    }
}

__global__
void loop(cuda_hvi_graph *d_g, int *d_degs, int *d_k, int *d_buffer, int *d_buffer_size, int *d_new_buffer, int *d_new_buffer_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *d_buffer_size) {
        int u = d_buffer[tid];
        for (int i = d_g->d_HVIOffset[u]; i < d_g->d_HVIOffset[u+1]; i++) {
            HybridVertexInterval hvi = d_g->d_HVIList[i];
            if (hvi & HVI_LEFT_BOUNDARY_MASK) {
                int left = hvi & GETNODE_MASK;
                i++;
                int right = d_g->d_HVIList[i] & GETNODE_MASK;
                for (int v = left; v <= right; v++) {
                    if (d_degs[v] > *d_k) {
                        int deg_v = atomicSub(d_degs + v, 1);
                        if (deg_v == *d_k + 1) {
                            int pos = atomicAdd(d_new_buffer_size, 1);
                            d_new_buffer[pos] = v;
                        }
                        if (deg_v <= *d_k) atomicAdd(d_degs + v, 1);
                    }
                }
            }
            else {
                int v = hvi;
                if (d_degs[v] > *d_k) {
                    int deg_v = atomicSub(d_degs + v, 1);
                    if (deg_v == *d_k + 1) {
                        int pos = atomicAdd(d_new_buffer_size, 1);
                        d_new_buffer[pos] = v;
                    }
                    if (deg_v <= *d_k) atomicAdd(d_degs + v, 1);
                }
            } 
        }
    }
}

void kcoreGPU(cuda_csr_graph g, vector<int> &core_numbers) {
    const int n = g.num_nodes;
    const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
    cuda_csr_graph *d_g;
    cudaMalloc((void **) &d_g, sizeof(cuda_csr_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_csr_graph), cudaMemcpyHostToDevice);

    int new_buffer_size, done_cnt = 0;
    int *d_degs, *d_buffer, *d_buffer_size, *d_new_buffer, *d_new_buffer_size, *d_k, *d_min;
    cudaMalloc((void **) &d_degs, n * sizeof(int));
    cudaMalloc((void **) &d_buffer, n * sizeof(int));
    cudaMalloc((void **) &d_buffer_size, sizeof(int));
    cudaMalloc((void **) &d_new_buffer, n * sizeof(int));
    cudaMalloc((void **) &d_new_buffer_size, sizeof(int));
    cudaMalloc((void **) &d_k, sizeof(int));
    cudaMalloc((void **) &d_min, sizeof(int));

    // printf("getting degrees\n");
    cudaMemcpy(d_degs, core_numbers.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    errorCheck(cudaDeviceSynchronize(), "getting degrees");
    // printf("done getting degrees\n");

    cudaMemset(d_buffer, 0, n * sizeof(int));
    cudaMemset(d_new_buffer, 0, n * sizeof(int));
    cudaMemset(d_buffer_size, 0, sizeof(int));
    cudaMemset(d_new_buffer_size, 0, sizeof(int));

    dim3 dimGridScan(n_blocks);
    dim3 dimBlock(N_THREADS_PER_BLOCK);
    for (int k = 0; k < n; k++) {
        if (done_cnt == n) break;
        cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_min, &n, sizeof(int), cudaMemcpyHostToDevice);
        find_min_if_not_less<<<dimGridScan, dimBlock>>>(n, d_k, d_degs, d_min);
        errorCheck(cudaDeviceSynchronize(), "finding k");
        cudaMemcpy(&k, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        if (k >= n) break;
        cudaMemcpy(d_k, d_min, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_new_buffer_size, 0, sizeof(int));
        scan<<<dimGridScan, dimBlock>>>(d_g, d_degs, d_new_buffer, d_new_buffer_size, d_k);
        errorCheck(cudaDeviceSynchronize(), "scanning");
        cudaMemcpy(&new_buffer_size, d_new_buffer_size, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("k: %d, new_buffer_size: %d\n", k, new_buffer_size);

        while (new_buffer_size > 0) {
            done_cnt += new_buffer_size;
            swap(d_buffer, d_new_buffer);
            swap(d_buffer_size, d_new_buffer_size);
            cudaMemset(d_new_buffer_size, 0, sizeof(int));
            const int l_blocks = (new_buffer_size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
            dim3 dimGridLoop(l_blocks);
            loop<<<dimGridLoop, dimBlock>>>(d_g, d_degs, d_k, d_buffer, d_buffer_size, d_new_buffer, d_new_buffer_size);
            errorCheck(cudaDeviceSynchronize(), "looping");
            cudaMemcpy(&new_buffer_size, d_new_buffer_size, sizeof(int), cudaMemcpyDeviceToHost);
            // printf("----new_buffer_size: %d\n", new_buffer_size);
        }
    }

    cudaMemcpy(core_numbers.data(), d_degs, n * sizeof(int), cudaMemcpyDeviceToHost);
}

void kcoreGPU(cuda_hvi_graph g, vector<int> &core_numbers) {
    const int n = g.num_nodes;
    const int n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
    cuda_hvi_graph *d_g;
    cudaMalloc((void **) &d_g, sizeof(cuda_hvi_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_hvi_graph), cudaMemcpyHostToDevice);

    int new_buffer_size, done_cnt = 0;
    int *d_degs, *d_buffer, *d_buffer_size, *d_new_buffer, *d_new_buffer_size, *d_k, *d_min;
    cudaMalloc((void **) &d_degs, n * sizeof(int));
    cudaMalloc((void **) &d_buffer, n * sizeof(int));
    cudaMalloc((void **) &d_buffer_size, sizeof(int));
    cudaMalloc((void **) &d_new_buffer, n * sizeof(int));
    cudaMalloc((void **) &d_new_buffer_size, sizeof(int));
    cudaMalloc((void **) &d_k, sizeof(int));
    cudaMalloc((void **) &d_min, sizeof(int));

    // cudaMemset(d_degs, 0, n * sizeof(int));
    cudaMemset(d_buffer, 0, n * sizeof(int));
    cudaMemset(d_new_buffer, 0, n * sizeof(int));
    cudaMemset(d_buffer_size, 0, sizeof(int));
    cudaMemset(d_new_buffer_size, 0, sizeof(int));

    dim3 dimGridScan(n_blocks);
    dim3 dimBlock(N_THREADS_PER_BLOCK);
    cudaMemcpy(d_degs, core_numbers.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    // get_degrees<<<dimGridScan, dimBlock>>>(d_g, d_degs);
    // errorCheck(cudaDeviceSynchronize(), "getting degrees");

    for (int k = 0; k < n; k++) {
        if (done_cnt == n) break;
        cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_min, &n, sizeof(int), cudaMemcpyHostToDevice);
        find_min_if_not_less<<<dimGridScan, dimBlock>>>(n, d_k, d_degs, d_min);
        errorCheck(cudaDeviceSynchronize(), "finding k");
        cudaMemcpy(&k, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        if (k >= n) break;
        cudaMemcpy(d_k, d_min, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_new_buffer_size, 0, sizeof(int));
        scan<<<dimGridScan, dimBlock>>>(d_g, d_degs, d_new_buffer, d_new_buffer_size, d_k);
        errorCheck(cudaDeviceSynchronize(), "scanning");
        cudaMemcpy(&new_buffer_size, d_new_buffer_size, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("k: %d, new_buffer_size: %d\n", k, new_buffer_size);

        while (new_buffer_size > 0) {
            done_cnt += new_buffer_size;
            swap(d_buffer, d_new_buffer);
            swap(d_buffer_size, d_new_buffer_size);
            cudaMemset(d_new_buffer_size, 0, sizeof(int));
            const int l_blocks = (new_buffer_size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
            dim3 dimGridLoop(l_blocks);
            loop<<<dimGridLoop, dimBlock>>>(d_g, d_degs, d_k, d_buffer, d_buffer_size, d_new_buffer, d_new_buffer_size);
            errorCheck(cudaDeviceSynchronize(), "looping");
            cudaMemcpy(&new_buffer_size, d_new_buffer_size, sizeof(int), cudaMemcpyDeviceToHost);
            // printf("----new_buffer_size: %d\n", new_buffer_size);
        }
    }

    cudaMemcpy(core_numbers.data(), d_degs, n * sizeof(int), cudaMemcpyDeviceToHost);
}