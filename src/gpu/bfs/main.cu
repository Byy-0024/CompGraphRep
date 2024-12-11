#include <iostream>
#include <chrono>

#include "../../Graph.hpp"
#include "../../Interval.hpp"
#include "bfs.cuh"

using namespace std;


void print(vector<int> &v);


class Checker {
	vector<int> expected_answer;
public:
	Checker(vector<int> exp_ans): expected_answer(exp_ans) {}

	pair<int, int> count_visited_vertices(const vector<int> &distance) {
		int depth = 0;
		int count = 0;
		for (int x : distance) {
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

void read_node_queries(string inputpath, vector<node> &queries) {
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
    printf("Has read %lu node queries from %s.\n", queries.size(), inputpath.c_str());
}

void cuda_bfs_test(string graphdict, string inputdict) {
    size_t pass_cnt_hvi = 0;
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
    _t.print("graph construnction");
    _size = G_comp.size_in_bytes();
    printf("Size of CompressGraph: %.3f MB!\n", (float)_size / (1 << 20));

    _t.clean();
    _t.start();
    HVI G_hvi(graphdict + "/origin/hvi_offsets.bin", graphdict + "/origin/hvi_list.bin");
    _t.stop();
    _t.print("hvi graph construnction");
    _size = G_hvi.size_in_bytes();
    printf("Size of new HVI Graph: %.3f MB!\n", (float)_size / (1 << 20));

    vector<node> node_queries;
    vector<int> distance;
    read_node_queries(inputdict, node_queries);
    time_counter _t_adj, _t_comp, _t_hvi;

    _t_adj.start();
    cuda_csr_graph G_cuda;
    G_cuda.init(G);
    _t_adj.stop();
    _t_comp.start();
    cuda_csr_graph G_cuda_comp;
    G_cuda_comp.init(G_comp);
    _t_comp.stop();
    _t_hvi.start();
    cuda_hvi_graph G_cuda_hvi;
    G_cuda_hvi.init(G_hvi);
    _t_hvi.stop();

    for (auto u : node_queries) {
        int startVertex = u % G.get_num_nodes();
        // int startVertex = node_queries[0];

        // Quadratic GPU BFS using CSR graph format.
        distance.clear();
        _t_adj.start();
        // cuda_csr_graph G_cuda;
        // G_cuda.init(G);
        bfsGPUQuadratic(startVertex, G_cuda, distance);
        // G_cuda.free();
        _t_adj.stop();
        Checker checker(distance);

        // Quadratic GPU BFS using CompressGraph format.
        distance.clear();
        _t_comp.start();
        // cuda_csr_graph G_cuda;
        // G_cuda.init(G);
        bfsGPUQuadratic(startVertex, G_cuda_comp, distance);
        // G_cuda.free();
        _t_comp.stop();

        // Quadratic GPU BFS using HVI graph format. 
        distance.clear();
        _t_hvi.start();
        // cuda_hvi_graph G_cuda_hvi;
        // G_cuda_hvi.init(G_hvi);
        bfsGPUQuadratic(startVertex, G_cuda_hvi, distance);
        // G_cuda_hvi.free();
        _t_hvi.stop();
        if (checker.check(distance)) pass_cnt_hvi++;
        else {
            printf("(HVI) Start vertex: %u.\n", startVertex);
            break;
        }
    }
    _t_adj.print("GPU BFS using CSR");
    _t_comp.print("GPU BFS using CompressGraph");
    _t_hvi.print("GPU BFS using HVI");
    printf("#Quadratic BFS verification pass: %lu.\n", pass_cnt_hvi);
    G_cuda.free();
    G_cuda_comp.free();
    G_cuda_hvi.free();
    
    // G_cuda.free();
    // G_cuda_hvi.free();
}

void cuda_bfs_test(string graphdict, string orderingmethod, string inputdict) {
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

    vector<node> node_queries;
    read_node_queries(inputdict, node_queries);
    vector<node> origin_to_new;
    if (orderingmethod != "origin") read_ordering(graphdict + "/" + orderingmethod + "/origin2new.bin", origin_to_new);

    vector<int> distance;
    size_t cuda_hvi_bfs_verification_pass_cnt = 0;
    time_counter _t_adj, _t_hvi;

    for (auto u : node_queries) {
        u %= G.get_num_nodes();
        int startVertex = orderingmethod == "origin" ? u : origin_to_new[u];

        // Quadratic GPU BFS using CSR graph format.
        distance = vector<int>(G.get_num_nodes());
        // _t_adj.start();
        cuda_csr_graph G_cuda;
        G_cuda.init(G);
        _t_adj.start();
        bfsGPUQuadratic(startVertex, G_cuda, distance);
        _t_adj.stop();
        G_cuda.free();
        // _t_adj.stop();
        Checker checker(distance);

        // Quadratic GPU BFS using hybrid CSR run graph format. 
        distance = vector<int>(G.get_num_nodes());
        // _t_hvi.start();
        cuda_hvi_graph G_cuda_hvi;
        G_cuda_hvi.init(G_hvi);
        _t_hvi.start();
        bfsGPUQuadratic(startVertex, G_cuda_hvi, distance);
        _t_hvi.stop();
        G_cuda_hvi.free();
        // _t_hvi.stop();
        if (checker.check(distance)) cuda_hvi_bfs_verification_pass_cnt++;
        else {
            printf("Start vertex: %u.\n", startVertex);
            break;
        }
    }
    _t_adj.print("GPU BFS using CSR");
    _t_hvi.print("GPU BFS using HVI");
    printf("#GPU BFS verification pass: %lu.\n", cuda_hvi_bfs_verification_pass_cnt);
    // G_cuda.free();
    // G_cuda_hvi.free();
}

// Tests speed of a BFS algorithm
int main(int argc, char **argv) {  
	string graphdict = argv[1];
    string orderingmethod = argv[2];
    string nodequerydict = argv[3];
    int device_id = stoi(argv[4]);
    cudaSetDevice(device_id);
    printf("======= %s Ordering ===========\n", orderingmethod.c_str());
    if (orderingmethod == "origin") cuda_bfs_test(graphdict, nodequerydict);
    else cuda_bfs_test(graphdict, orderingmethod, nodequerydict);
    // printf("======= Greedy Ordering ===========\n");
    // cuda_bfs_test(graphdict, orderingdict, nodequerydict);
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
