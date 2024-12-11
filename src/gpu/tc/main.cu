#include <iostream>
#include <chrono>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "tc.cuh"

using namespace std;

void read_ordering(string orderingdict, vector<node> &origin_to_new, int num_nodes) {
    ifstream input(orderingdict);
    string line;
    origin_to_new.clear();
    origin_to_new.resize(num_nodes, UINT_MAX);
    while (getline(input, line)) {
        int u, v;
        stringstream(line).ignore(0, ' ') >> u >> v;
        origin_to_new[u] = v;
    }
}

void cuda_tc_test(string graphdict) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/origin/csr_vlist.bin", graphdict + "/origin/csr_elist.bin");
    _t.stop();
    _t.print("graph construnction");
    HVI G_hvi(G);
    size_t _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));
    
    ull res;
    time_counter _t_csr, _t_hvi;

    _t_csr.start();
    cuda_coo_graph G_cuda;
    G_cuda.init(G);
    // _t_csr.start();
    tcGPU(G_cuda, res);
    // _t_csr.stop();
    G_cuda.free();
    _t_csr.stop();
    _t_csr.print("GPU triangle counting using CSR");
    printf("#Triangles: %llu.\n", res);

    _t_hvi.start();
    cuda_hvi_coo_graph G_cuda_hvi;
    G_cuda_hvi.init(G_hvi);
    // _t_hvi.start();
    tcGPU(G_cuda_hvi, res);
    // _t_hvi.stop();
    G_cuda_hvi.free();
    _t_hvi.stop();
    _t_hvi.print("GPU triangle counting using HVI");
    printf("#Triangles: %llu.\n", res);
}

void cuda_tc_test(string graphdict, string orderingmethod) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/" + orderingmethod + "/csr_vlist.bin", graphdict + "/" + orderingmethod + "/csr_elist.bin");
    _t.stop();
    _t.print("csr graph construnction");
    size_t _size = G.size_in_bytes();
    printf("Size of new CSR Graph: %.3f MB!\n", (float)_size / (1 << 20));

    _t.clean();
    _t.start();
    HVI G_hvi(graphdict + "/" + orderingmethod + "/hvi_offsets.bin", graphdict + "/" + orderingmethod + "/hvi_list.bin");
    _t.stop();
    _t.print("hvi graph construnction");
    _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));

    time_counter _t_csr, _t_hvi;
    ull res = 0;

    _t_csr.start();
    cuda_coo_graph G_cuda;
    G_cuda.init(G);
    tcGPU(G_cuda, res);
    G_cuda.free();
    _t_csr.stop();
    _t_csr.print("GPU triangle counting using CSR");
    printf("#Triangles: %llu.\n", res);

    _t_hvi.start();
    cuda_hvi_coo_graph G_cuda_hvi;
    G_cuda_hvi.init(G_hvi);
    tcGPU(G_cuda_hvi, res);
    G_cuda_hvi.free();
    _t_hvi.stop();
    _t_hvi.print("GPU triangle counting using HVI");
    printf("#Triangles: %llu.\n", res);
}

// Tests speed of a BFS algorithm
int main(int argc, char **argv) {  
	string graphdict = argv[1];
    string orderingmethod = argv[2];
    int device_id = stoi(argv[3]);
    cudaSetDevice(device_id);
    printf("======= %s Ordering ===========\n", orderingmethod.c_str());
    cuda_tc_test(graphdict, orderingmethod);
    // printf("======= Greedy Ordering ===========\n");
    // cuda_bfs_test(graphdict, orderingdict, nodequerydict);
	return 0;
}
