#include <iostream>
#include <chrono>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "ee.cuh"

using namespace std;


void print(vector<int> &v);

void read_ordering(string orderingdict, vector<node> &origin_to_new) {
    ifstream input(orderingdict);
    if (!input.is_open()) {
        cerr << "Unable to open ordering binfile" << endl;
        return;
    }
    input.seekg(0, ios::end);
    streamsize size = input.tellg();
    origin_to_new.resize(size / sizeof(int));
    input.seekg(0, ios::beg);
    input.read(reinterpret_cast<char *>(origin_to_new.data()), size);
    input.close();
    printf("Has read ordering from %s, origin_to_new.size(): %lu.\n", orderingdict.c_str(), origin_to_new.size());
}

void read_edge_queries(string inputpath, vector<node> &queries) {
    ifstream query_binfile(inputpath, ios::binary);
    if (!query_binfile.is_open()) {
        cerr << "Unable to open csr vlist binfile" << endl;
        return;
    }
    query_binfile.seekg(0, ios::end);
    streamsize size = query_binfile.tellg();
    queries.resize(size / sizeof(int));
    query_binfile.seekg(0, ios::beg);
    query_binfile.read(reinterpret_cast<char *>(queries.data()), size);
    query_binfile.close();
    printf("Has read %lu node queries from %s.\n", queries.size() >> 1, inputpath.c_str());
}

void cuda_ee_test(string graphdict, string inputdict) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/origin/csr_vlist.bin", graphdict + "/origin/csr_elist.bin");
    _t.stop();
    _t.print("graph construnction");
    HVI G_hvi(G);
    size_t _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));

    int res_csr = 0, res_hvi = 0;
    vector<node> edge_queries;
    read_edge_queries(inputdict, edge_queries);
    time_counter _t_adj, _t_hvi;

    _t_adj.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    eeGPU(G_cuda, edge_queries, res_csr);
    G_cuda.free();
    _t_adj.stop();
    _t_adj.print("GPU edge existance test using CSR");
    printf("#Edges: %d\n", res_csr);

    _t_hvi.start();
    cuda_hvi_graph G_cuda_hvi;
    G_cuda_hvi.init(G_hvi);
    eeGPU(G_cuda_hvi, edge_queries, res_hvi);
    G_cuda_hvi.free();
    _t_hvi.stop();
    _t_hvi.print("GPU edge existance test using HVI");
    printf("#Edges: %d\n", res_hvi);
}

void cuda_ee_test(string graphdict, string orderingmethod, string inputdict) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/" + orderingmethod + "/csr_vlist.bin", graphdict + "/" + orderingmethod + "/csr_elist.bin");
    _t.stop();
    _t.print("csr graph construnction");
    size_t _size = G.size_in_bytes();
    printf("Size of new CSR Graph: %.3f MB!\n", (float)_size / (1 << 20));

    vector<node> edge_queries;
    read_edge_queries(inputdict, edge_queries);

    vector<node> origin_to_new;
    if (orderingmethod != "origin") {
        read_ordering(graphdict + "/" + orderingmethod + "/origin2new.bin", origin_to_new);
    }
    const int n = G.get_num_nodes();
    for (size_t i = 0; i < edge_queries.size(); i++) {
        edge_queries[i] %= n;
        if (orderingmethod != "origin") {
            edge_queries[i] = origin_to_new[edge_queries[i]];
        }
    }

    int res_csr = 0, res_hvi = 0;
    time_counter _t_adj, _t_hvi;
    _t_adj.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    eeGPU(G_cuda, edge_queries, res_csr);
    G_cuda.free();
    _t_adj.stop();
    _t_adj.print("GPU edge existance test using CSR");
    printf("#Edges: %d\n", res_csr);

    _t.clean();
    _t.start();
    HVI G_hvi(graphdict + "/" + orderingmethod + "/hvi_offsets.bin", graphdict + "/" + orderingmethod + "/hvi_list.bin");
    _t.stop();
    _t.print("hvi graph construnction");
    _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));
    _t_hvi.start();
    cuda_hvi_graph G_cuda_hvi;
    G_cuda_hvi.init(G_hvi);
    eeGPU(G_cuda_hvi, edge_queries, res_hvi);
    G_cuda_hvi.free();
    _t_hvi.stop();
    _t_hvi.print("GPU edge existance test using HVI");
    printf("#Edges: %d\n", res_hvi);

    if (orderingmethod == "origin") {
        time_counter _t_comp;
        int res_comp = 0;
        _t.clean();
        _t.start();
        Graph G_comp(graphdict + "/compress/csr_vlist.bin", graphdict + "/compress/csr_elist.bin");
        _t.stop();
        _t.print("CompressGraph construnction");
        _size = G_comp.size_in_bytes();
        printf("Size of CompressGraph: %.3f MB!\n", (float)_size / (1 << 20));
        _t_comp.start();
        cuda_csr_graph G_cuda_comp;
        G_cuda_comp.init(G_comp);
        eeGPU(G_cuda_comp, edge_queries, res_comp);
        G_cuda_comp.free();
        _t_comp.stop();
        _t_comp.print("GPU edge existance test using CompressGraph");
        // printf("#Edges: %d\n", res_hvi);
    }
}

int main(int argc, char **argv) {  
	string graphdict = argv[1];
    string orderingmethod = argv[2];
    string nodequerydict = argv[3];
    int device_id = stoi(argv[4]);
    cudaSetDevice(device_id);
    printf("======= %s Ordering ===========\n", orderingmethod.c_str());
    cuda_ee_test(graphdict, orderingmethod, nodequerydict);
	return 0;
}


void print(vector<int> &v) {
	cout << "{ ";
	for (int i = 0; i < v.size(); ++i) {
		cout << v[i];
		if (i < v.size() - 1)
			cout << ", ";
	}
	cout << " }" << endl;
}
