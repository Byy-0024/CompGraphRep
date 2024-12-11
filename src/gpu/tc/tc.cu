#include "tc.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 10)

__global__
void tc_kernel(cuda_coo_graph *d_g, ull *res) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ ull sdata[N_THREADS_PER_BLOCK];  
    ull t_res = 0;
    if (tid < d_g->num_edges) {
        int u = d_g->d_sourceNodesList[tid];
        int v = d_g->d_targetNodesList[tid];
        cnt_intersection(u, v, d_g->d_targetNodesList, d_g->d_edgesOffset, d_g->d_edgesSize, &t_res);
    }
    else {
        t_res = 0;
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

// __global__
// void tc_kernel(cuda_hvi_coo_graph *d_g, ull *res) {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     __shared__ ull sdata[N_THREADS_PER_BLOCK]; 
//     ull t_res = 0;
//     if (tid < d_g->num_hvi) {
//         int u = d_g->d_sourceNodesList[tid];
//         HybridVertexInterval v = d_g->d_HVIList[tid];
//         if (v & HVI_LEFT_BOUNDARY_MASK) {
//             int left = v & GETNODE_MASK;
//             int right = d_g->d_HVIList[tid+1] & GETNODE_MASK;
//             for (int i = left; i < (left + right) / 2; ++i) {
//                 cnt_intersection_hvi(u, i, d_g->d_HVIList, d_g->d_HVIOffset, &t_res);
//             }
//         }
//         else if (v & HVI_RIGHT_BOUNDARY_MASK) {
//             int left = d_g->d_HVIList[tid-1] & GETNODE_MASK;
//             int right = v & GETNODE_MASK;
//             for (int i = (left + right) / 2; i <= right; ++i) {
//                 cnt_intersection_hvi(u, i, d_g->d_HVIList, d_g->d_HVIOffset, &t_res);
//             }
//         }
//         else {
//             cnt_intersection_hvi(u, v, d_g->d_HVIList, d_g->d_HVIOffset, &t_res);   
//         }
//     }
//     else {
//         t_res = 0;
//     }

//     int local_id = threadIdx.x;
//     sdata[local_id] = t_res;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (local_id < s) {
//             sdata[local_id] += sdata[local_id + s];
//         }
//         __syncthreads();
//     }
//     if (local_id == 0) {
//         atomicAdd(res, sdata[0]);
//     }
// }

__global__ 
void tc_kernel(cuda_hvi_coo_graph *d_g, ull *res, node *u_buffer, node *v_buffer, int buffer_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ ull sdata[N_THREADS_PER_BLOCK];  
    ull t_res = 0;
    if (tid < d_g->num_edges) {
        node u = u_buffer[tid];
        node v = v_buffer[tid];
        if (u != v) cnt_intersection_hvi(u, v, d_g->d_HVIList, d_g->d_HVIOffset, &t_res);
    }
    else{
        t_res = 0;
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
void compute_buffer_sizes(cuda_hvi_coo_graph *d_g, int *d_buffer_sizes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_g->num_hvi) {
        HybridVertexInterval v = d_g->d_HVIList[tid];
        d_buffer_sizes[tid] = (v & HVI_LEFT_BOUNDARY_MASK) ? (d_g->d_HVIList[tid+1] & GETNODE_MASK) - (v & GETNODE_MASK) : 1;
    }
}


__global__
void devide_work(cuda_hvi_coo_graph *d_g, node *u_buffer, node *v_buffer, int *buffer_offsets) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_g->num_hvi) {
        node u = d_g->d_sourceNodesList[tid];
        node v = (d_g->d_HVIList[tid] & GETNODE_MASK);
        for (int i = buffer_offsets[tid]; i < buffer_offsets[tid + 1]; ++i) { 
            u_buffer[i] = u;
            v_buffer[i] = v + (i - buffer_offsets[tid]);
        }
    }
}

void tcGPU(cuda_coo_graph g, ull &res) {
    const int m = g.num_edges;
    ull n_blocks = (m + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK; // ceil(num_edges / THREADS_PER_BLOCK)

    cuda_coo_graph *d_g;
    cudaMalloc((void **)&d_g, sizeof(cuda_coo_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_coo_graph), cudaMemcpyHostToDevice);
    
    ull *d_res;
    cudaMalloc((void **)&d_res, sizeof(ull));
    cudaMemset(d_res, 0, sizeof(ull));

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(N_THREADS_PER_BLOCK);
    
    tc_kernel<<<dimGrid, dimBlock>>>(d_g, d_res);
    errorCheck(cudaDeviceSynchronize(), "counting triangles");
    cudaMemcpy(&res, d_res, sizeof(ull), cudaMemcpyDeviceToHost);
    cudaFree(d_res);
}

void tcGPU(cuda_hvi_coo_graph g, ull &res) {
    const int n = g.num_hvi;
    const int m = g.num_edges;
    // printf("n = %d, m = %d\n", n, m);
    ull n_blocks = (n + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK; 
    ull m_blocks = (m + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    cuda_hvi_coo_graph *d_g;
    cudaMalloc((void **)&d_g, sizeof(cuda_hvi_coo_graph));
    cudaMemcpy(d_g, &g, sizeof(cuda_hvi_coo_graph), cudaMemcpyHostToDevice);
    
    thrust::device_vector<int> d_buffer_sizes(n);
    compute_buffer_sizes<<<n_blocks, N_THREADS_PER_BLOCK>>>(d_g, thrust::raw_pointer_cast(d_buffer_sizes.data()));
    errorCheck(cudaDeviceSynchronize(), "computing buffer sizes");
    thrust::host_vector<int> buffer_sizes(n);
    thrust::copy(d_buffer_sizes.begin(), d_buffer_sizes.end(), buffer_sizes.begin());

    thrust::host_vector<int> buffer_offsets(n+1);
    thrust::device_vector<int> d_buffer_offsets(n+1);
    thrust::exclusive_scan(buffer_sizes.begin(), buffer_sizes.end(), buffer_offsets.begin());
    buffer_offsets[n] = m;
    thrust::copy(buffer_offsets.begin(), buffer_offsets.end(), d_buffer_offsets.begin());
    // for (int i = 1; i < n; ++i) {
    //     if (buffer_offsets[i] == 0) {
    //         printf("Unexpected buffer_offsets[%d] = 0\n", i);
    //         printf("buffer_offsets[%d] = %d\n", i-1, buffer_offsets[i-1]);
    //         printf("buffer_sizes[%d] = %d\n", i-1, buffer_sizes[i-1]);
    //         break;
    //     }
    // }

    node *d_u_buffer, *d_v_buffer;
    cudaMalloc((void **)&d_u_buffer, m * sizeof(node));
    cudaMalloc((void **)&d_v_buffer, m * sizeof(node));
    devide_work<<<n_blocks, N_THREADS_PER_BLOCK>>>(d_g, d_u_buffer, d_v_buffer, thrust::raw_pointer_cast(d_buffer_offsets.data()));
    errorCheck(cudaDeviceSynchronize(), "deviding works");
    
    ull *d_res;
    cudaMalloc((void **)&d_res, sizeof(ull));
    cudaMemset(d_res, 0, sizeof(ull));
    
    tc_kernel<<<m_blocks, N_THREADS_PER_BLOCK>>>(d_g, d_res, d_u_buffer, d_v_buffer, m);
    errorCheck(cudaDeviceSynchronize(), "counting triangles");
    cudaMemcpy(&res, d_res, sizeof(ull), cudaMemcpyDeviceToHost);
    cudaFree(d_res);
}