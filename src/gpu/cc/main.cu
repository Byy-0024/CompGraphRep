#include <iostream>
#include <chrono>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "cc.cuh"

using namespace std;

void print(vector<int> &v, int print_size) {
	cout << "{ ";
	for (int i = 0; i < print_size; ++i) {
		cout << v[i];
		if (i < print_size - 1)
			cout << ", ";
	}
	cout << "... }" << endl;
}

class Checker {
	vector<int> expected_answer;
public:
	Checker(vector<int> exp_ans): expected_answer(exp_ans) {}

	pair<int, int> count_visited_vertices(const vector<int> &component_ids) {
		int depth = 0;
		int count = 0;
		for (int x : component_ids) {
			if (x < INT_MAX) {
				++count;
				if (x > depth) {
					depth = x;
				}
			}
		}
		return {count, depth};
	}

	bool check(vector<int> answer) {
		assert(answer.size() == expected_answer.size());
		bool is_ok = true;
		int position_wrong = -1;
		for (int i = 0; i < answer.size(); ++i) {
			if (answer.at(i) != expected_answer.at(i)) {
				is_ok = false;
				position_wrong = i;
				break;
			}
		}
        // return is_ok;
		if (is_ok) {
			// pair<int, int> graph_output = count_visited_vertices(answer);
			// int n_visited_vertices = graph_output.first;
			// int depth = graph_output.second;
			// printf("CHECKED SUCCESSFULY! Number of visited vertices: %i, depth: %i \n", n_visited_vertices, depth);
            return true;
		}
		else {
			printf("Something went wrong!\n");
			printf("Answer at %i equals %i but should be equal to %i\n", position_wrong, answer[position_wrong], expected_answer[position_wrong]);
            return false;
		}
	}
};

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

void cuda_cc_test(string graphdict) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/origin/csr_vlist.bin", graphdict + "/origin/csr_elist.bin");
    _t.stop();
    _t.print("graph construnction");
    size_t _size = G.size_in_bytes();
    printf("Size of CSR Graph: %.3f MB!\n", (float)_size / (1 << 20));

    _t.clean();
    _t.start();
    Graph G_comp(graphdict + "/compress/csr_vlist.bin", graphdict + "/compress/csr_elist.bin");
    _t.stop();
    _t.print("CompressGraph construnction");
    _size = G_comp.size_in_bytes();
    printf("Size of CompressGraph: %.3f MB!\n", (float)_size / (1 << 20));

    _t.clean();
    _t.start();
    HVI G_hvi(graphdict + "/origin/hvi_offsets.bin", graphdict + "/origin/hvi_list.bin");
    _t.stop();
    _t.print("hvi graph construnction");
    _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));
    
    vector<int> component_ids;
    time_counter _t_csr, _t_comp, _t_hvi;

    // Shiloach-Vishkin CC using CSR graph format.
    component_ids.clear();
    _t_csr.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    ccShiloachVishkin(G_cuda, component_ids);
    G_cuda.free();
    _t_csr.stop();
    Checker checker(component_ids);
    _t_csr.print("GPU CC using CSR");
    print(component_ids, 10);

    // Shiloach-Vishkin CC using CompressGraph format. 
    component_ids.clear();
    _t_comp.start();
    cuda_csr_graph G_cuda_comp;
    G_cuda_comp.init(G_comp);
    ccShiloachVishkin(G_cuda_comp, component_ids);
    G_cuda_comp.free();
    _t_comp.stop();
    _t_comp.print("GPU BFS using CompressGraph");
    print(component_ids, 10);
    if (checker.check(component_ids)) printf("Connected component verification: Pass!\n");
    else printf("Connected component verification: Fail!\n");

    // Shiloach-Vishkin CC using HVI graph format. 
    component_ids.clear();
    _t_hvi.start();
    cuda_hvi_graph G_cuda_run;
    G_cuda_run.init(G_hvi);
    ccShiloachVishkin(G_cuda_run, component_ids);
    G_cuda_run.free();
    _t_hvi.stop();
    _t_hvi.print("GPU BFS using HVI");
    print(component_ids, 10);
    if (checker.check(component_ids)) printf("Connected component verification: Pass!\n");
    else printf("Connected component verification: Fail!\n");
}

void cuda_cc_test(string graphdict, string orderingmethod) {
    vector<int> component_ids;
    time_counter _t_csr, _t_comp, _t_hvi;
    time_counter _t;

    // Shiloach-Vishkin CC using CSR graph format.
    _t.start();
    Graph G(graphdict + "/" + orderingmethod + "/csr_vlist.bin", graphdict + "/" + orderingmethod + "/csr_elist.bin");
    _t.stop();
    _t.print("csr graph construnction");
    size_t _size = G.size_in_bytes();
    printf("Size of new CSR Graph: %.3f MB!\n", (float)_size / (1 << 20));

    _t_csr.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    ccShiloachVishkin(G_cuda, component_ids);
    G_cuda.free();
    _t_csr.stop();
    Checker checker(component_ids);
    _t_csr.print("GPU CC using CSR");
    print(component_ids, 10);

    // Shiloach-Vishkin CC using CompressGraph format. 
    if (orderingmethod == "origin") {
        _t.clean();
        _t.start();
        Graph G_comp(graphdict + "/compress/csr_vlist.bin", graphdict + "/compress/csr_elist.bin");
        _t.stop();
        _t.print("CompressGraph construnction");
        _size = G_comp.size_in_bytes();
        printf("Size of CompressGraph: %.3f MB!\n", (float)_size / (1 << 20));

        component_ids.clear();
        _t_comp.start();
        cuda_csr_graph G_cuda_comp;
        G_cuda_comp.init(G_comp);
        ccShiloachVishkin(G_cuda_comp, component_ids);
        G_cuda_comp.free();
        _t_comp.stop();
        _t_comp.print("GPU CC using CompressGraph");
        component_ids.erase(component_ids.begin() + G.get_num_nodes(), component_ids.end());
        // print(component_ids, 10);
        // if (checker.check(component_ids)) printf("Connected component verification: Pass!\n");
        // else printf("Connected component verification: Fail!\n");
    }

    // Shiloach-Vishkin CC using HVI graph format. 
    _t.clean();
    _t.start();
    HVI G_hvi(graphdict + "/" + orderingmethod + "/hvi_offsets.bin", graphdict + "/" + orderingmethod + "/hvi_list.bin");
    _t.stop();
    _t.print("hvi graph construnction");
    _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));

    component_ids.clear();
    _t_hvi.start();
    cuda_hvi_graph G_cuda_run;
    G_cuda_run.init(G_hvi);
    ccShiloachVishkin(G_cuda_run, component_ids);
    G_cuda_run.free();
    _t_hvi.stop();
    _t_hvi.print("GPU CC using HVI");
    print(component_ids, 10);
    if (checker.check(component_ids)) printf("Connected component verification: Pass!\n");
    else printf("Connected component verification: Fail!\n");
}

// Tests speed of a BFS algorithm
int main(int argc, char **argv) {  
	string graphdict = argv[1];
    string orderingmethod = argv[2];
    int device_id = stoi(argv[3]);
    cudaSetDevice(device_id);
    printf("======= %s Ordering ===========\n", orderingmethod.c_str());
    cuda_cc_test(graphdict, orderingmethod);
    // printf("======= Greedy Ordering ===========\n");
    // cuda_bfs_test(graphdict, orderingdict, nodequerydict);
	return 0;
}
