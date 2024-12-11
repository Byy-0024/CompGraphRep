#include <iostream>
#include <chrono>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "pr.cuh"

using namespace std;

template <class T>
void print(vector<T> &v, int print_size) {
	cout << "{ ";
	for (int i = 0; i < print_size; ++i) {
		cout << v[i];
		if (i < print_size - 1)
			cout << ", ";
	}
	cout << "... }" << endl;
}

// class Checker {
// 	vector<int> expected_answer;
// public:
// 	Checker(vector<int> exp_ans): expected_answer(exp_ans) {}

// 	pair<int, int> count_visited_vertices(const vector<int> &pr_vals) {
// 		int depth = 0;
// 		int count = 0;
// 		for (int x : pr_vals) {
// 			if (x < INT_MAX) {
// 				++count;
// 				if (x > depth) {
// 					depth = x;
// 				}
// 			}
// 		}
// 		return {count, depth};
// 	}

// 	bool check(vector<int> answer) {
// 		assert(answer.size() == expected_answer.size());
// 		bool is_ok = true;
// 		int position_wrong = -1;
// 		for (int i = 0; i < answer.size(); ++i) {
// 			if (answer.at(i) != expected_answer.at(i)) {
// 				is_ok = false;
// 				position_wrong = i;
// 				break;
// 			}
// 		}
//         // return is_ok;
// 		if (is_ok) {
// 			// pair<int, int> graph_output = count_visited_vertices(answer);
// 			// int n_visited_vertices = graph_output.first;
// 			// int depth = graph_output.second;
// 			// printf("CHECKED SUCCESSFULY! Number of visited vertices: %i, depth: %i \n", n_visited_vertices, depth);
//             return true;
// 		}
// 		else {
// 			printf("Something went wrong!\n");
// 			printf("Answer at %i equals %i but should be equal to %i\n", position_wrong, answer[position_wrong], expected_answer[position_wrong]);
//             return false;
// 		}
// 	}
// };

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

void cuda_pr_test(string graphdict) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/origin/csr_vlist.bin", graphdict + "/origin/csr_elist.bin");
    _t.stop();
    _t.print("graph construnction");
    HVI G_hvi(G);
    size_t _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));
    
    int n = G.get_num_nodes(), max_iter = 100;
    float epsilon = 0.05;
    vector<float> pr_vals_csr(n, 1), pr_vals_hvi(n, 1);
    
    time_counter _t_csr, _t_hvi;

    // _t_csr.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    _t_csr.start();
    pageRank(G_cuda, pr_vals_csr, max_iter, epsilon);
    _t_csr.stop();
    G_cuda.free();
    // _t_csr.stop();
    // Checker checker(pr_vals);
    _t_csr.print("GPU PageRank computing using CSR");
    print(pr_vals_csr, 10);

    // _t_hvi.start();
    cuda_hvi_graph G_cuda_run;
    G_cuda_run.init(G_hvi);
    _t_hvi.start();
    pageRank(G_cuda_run, pr_vals_hvi, max_iter, epsilon);
    _t_hvi.stop();
    G_cuda_run.free();
    // _t_hvi.stop();
    _t_hvi.print("GPU PageRank computing using HVI");
    print(pr_vals_hvi, 10);
    
    // if (checker.check(pr_vals)) printf("K-core decomposition verification: Pass!\n");
    // else printf("K-core decomposition verification: Fail!\n");
}

void cuda_pr_test(string graphdict, string orderingmethod) {
    time_counter _t;
    _t.start();
    Graph G(graphdict + "/" + orderingmethod + "/csr_vlist.bin", graphdict + "/" + orderingmethod + "/csr_elist.bin");
    _t.stop();
    _t.print("csr graph construnction");
    size_t _size = G.size_in_bytes();
    printf("Size of new CSR Graph: %.3f MB!\n", (float)_size / (1 << 20));
    
    float epsilon = 0.05;
    int n = G.get_num_nodes(), max_iter = 100;
    vector<float> pr_vals_csr(n, 1), pr_vals_hvi(n, 1);
    
    time_counter _t_csr, _t_hvi;
    _t_csr.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    pageRank(G_cuda, pr_vals_csr, max_iter, epsilon);
    G_cuda.free();
    _t_csr.stop();
    // Checker checker(pr_vals);
    _t_csr.print("GPU PageRank computing using CSR");
    print(pr_vals_csr, 10);

    _t.clean();
    _t.start();
    HVI G_hvi(graphdict + "/" + orderingmethod + "/hvi_offsets.bin", graphdict + "/" + orderingmethod + "/hvi_list.bin");
    _t.stop();
    _t.print("hvi graph construnction");
    _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    cuda_hvi_graph G_cuda_run;
    G_cuda_run.init(G_hvi);
    pageRank(G_cuda_run, pr_vals_hvi, max_iter, epsilon);
    G_cuda_run.free();
    _t_hvi.stop();
    _t_hvi.print("GPU PageRank computing using HVI");
    print(pr_vals_hvi, 10);

    if (orderingmethod == "origin") {
        _t.clean();
        _t.start();
        Graph G_comp(graphdict + "/compress/csr_vlist.bin", graphdict + "/compress/csr_elist.bin");
        _t.stop();
        _t.print("CompressGraph construnction");
        size_t _size = G_comp.size_in_bytes();
        printf("Size of CompressGraph: %.3f MB!\n", (float)_size / (1 << 20));

        vector<float> pr_vals_comp(G_comp.get_num_nodes(), 1);
        
        time_counter _t_comp;
        _t_comp.start();
        cuda_csr_graph G_cuda_comp;
        G_cuda_comp.init(G_comp);
        pageRank(G_cuda_comp, pr_vals_comp, max_iter, epsilon);
        G_cuda_comp.free();
        _t_comp.stop();
        _t_comp.print("GPU PageRank computing using CompressGraph");
    }
}

// Tests speed of a BFS algorithm
int main(int argc, char **argv) {  
	string graphdict = argv[1];
    string orderingmethod = argv[2];
    int device_id = stoi(argv[3]);
    cudaSetDevice(device_id);
    printf("======= %s Ordering ===========\n", orderingmethod.c_str());
    cuda_pr_test(graphdict, orderingmethod);
	return 0;
}
