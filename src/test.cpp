#include "Graph.hpp"
#include "CompressGraph.hpp"
#include "Interval.hpp"

using namespace std;

template <class T>
bool vec_fast_verification(const vector<T> &vec1, const vector<T> &vec2) {
    if (vec1.size() != vec2.size()) return false;
    for (size_t i = 0; i < vec1.size(); i++) {
        if (vec1[i] != vec2[i]) {
            cout << "Error index " << i << ", vec1 = " << vec1[i] << ", vec2 = " << vec2[i] << "." << endl;
            return false;
        }
    }
    return true;
}

template <class T>
void print_vec(vector<T> &v, int print_size) {
	cout << "{ ";
	for (int i = 0; i < print_size; ++i) {
		cout << v[i];
		if (i < print_size - 1)
			cout << ", ";
	}
	cout << "... }" << endl;
}

void generate_node_queries(size_t num_queries, string outputpath) {
    vector<node> queries;
    queries.reserve(num_queries);
    for (size_t i = 0; i < num_queries; i++) {
        queries.push_back(rand());
    }

    ofstream query_binfile(outputpath, ios::binary);
    query_binfile.write(reinterpret_cast<char*>(queries.data()), num_queries * sizeof(node));
    query_binfile.close();
    vector<node>().swap(queries);
}

void generate_edge_queries(size_t num_queries, string outputpath) {
    vector<node> queries;
    queries.resize(2 * num_queries);
    for (size_t i = 0; i < 2 * num_queries; i++) {
        queries[i] = rand();
    }

    ofstream query_binfile(outputpath, ios::binary);
    query_binfile.write(reinterpret_cast<char*>(queries.data()), 2 * num_queries * sizeof(node));
    query_binfile.close();
    vector<node>().swap(queries);
}

void generate_subgraph_query(int i, string outputpath) {
    vector<int> csr_vlist;
    vector<node> csr_elist;
    switch (i)
    {
    case 0:
        csr_vlist = {0,2,4,6};
        csr_elist = {1,2,0,2,0,1};
        break;
    case 1:
        csr_vlist = {0,2,4,6,8};
        csr_elist = {1,3,0,2,1,3,0,2};
        break;
    case 2:
        csr_vlist = {0,3,5,8,10};
        csr_elist = {1,2,3,0,2,0,1,3,0,2};
        break;    
    case 3:
        csr_vlist = {0,3,6,9,12};
        csr_elist = {1,2,3,0,2,3,0,1,3,0,1,2};
        break;  
    case 4:
        csr_vlist = {0,2,6,10,13,16};
        csr_elist = {1,2,0,2,3,4,0,1,3,4,1,2,4,1,2,3};
        break;  
    case 5:
        csr_vlist = {0,4,8,12,16,20};
        csr_elist = {1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3};
        break;  
    case 6:
        csr_vlist = {0,2,5,8,11,14};
        csr_elist = {1,2,0,3,4,0,3,4,1,2,4,1,2,3};
        break;  
    default:
        break;
    }

    ofstream csr_vlist_binfile(outputpath + "/q" + to_string(i) + "_csr_vlist.bin", ios::binary);
    csr_vlist_binfile.write(reinterpret_cast<char*>(csr_vlist.data()), csr_vlist.size() * sizeof(int));
    csr_vlist_binfile.close();

    ofstream csr_elist_binfile(outputpath + "/q" + to_string(i) + "_csr_elist.bin", ios::binary);
    csr_elist_binfile.write(reinterpret_cast<char*>(csr_elist.data()), csr_elist.size() * sizeof(node));
    csr_elist_binfile.close();
}

void read_ordering(string inputpath, vector<node> &origin_to_new) {
    ifstream ordering_binfile(inputpath, ios::binary);
    if (!ordering_binfile.is_open()) {
        cerr << "Unable to open ordering binfile from " << inputpath << " !" << endl;
        return;
    }
    ordering_binfile.seekg(0, ios::end);
    streamsize size = ordering_binfile.tellg();
    origin_to_new.resize(size / sizeof(int));
    ordering_binfile.seekg(0, ios::beg);
    ordering_binfile.read(reinterpret_cast<char *>(origin_to_new.data()), size);
    ordering_binfile.close();
    printf("Has read ordering from %s.\n", inputpath.c_str());
}

void read_node_queries(string inputpath, vector<node> &queries) {
    ifstream query_binfile(inputpath, ios::binary);
    if (!query_binfile.is_open()) {
        cerr << "Unable to open node query binfile" << endl;
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

void read_edge_queries(string inputpath, vector<node> &queries) {
    ifstream query_binfile(inputpath, ios::binary);
    if (!query_binfile.is_open()) {
        cerr << "Unable to open edge query binfile" << endl;
        return;
    }
    query_binfile.seekg(0, ios::end);
    streamsize size = query_binfile.tellg();
    queries.resize(size / sizeof(int));
    query_binfile.seekg(0, ios::beg);
    query_binfile.read(reinterpret_cast<char *>(queries.data()), size);
    query_binfile.close();
    printf("Has read %lu edge queries from %s.\n", queries.size() >> 1, inputpath.c_str());
}

void neis_test(const Graph &g, const CompressGraph &g_comp) {
    size_t neis_check_pass_cnt = 0, num_nodes = g.get_num_nodes();
    vector<node> neis1, neis2;
    node first_not_pass_node = INT_MAX;
    for (node i = 0; i < num_nodes; i++) {
        g.get_neis(i, neis1);
        // g_comp.get_neis(i, neis2);
        neis2.clear();
        for (CompressGraphIterator iter(g_comp, i); iter.curr_ptr != iter.end_ptr; ++iter) {
            neis2.push_back(iter.get_val());
        }
        if (vec_fast_verification(neis1, neis2)) neis_check_pass_cnt++;
        else if (first_not_pass_node == INT_MAX) first_not_pass_node = i;
    }
    printf("Number of vertices: %lu, pass neigbors verification: %lu.\n", num_nodes, neis_check_pass_cnt);
    if (first_not_pass_node != INT_MAX) {
        g.print_neis(first_not_pass_node);
        g_comp.print_neis(first_not_pass_node);
    }
}

void ee_test(string querypath, Graph &g, CompressGraph &g_comp, HVI &g_hvi, vector<node> &origin_to_new) {
    vector<node> queries;
    read_edge_queries(querypath, queries);
    size_t has_edge_cnt_csr = 0, has_edge_cnt_comp = 0, has_edge_cnt_hvi = 0, n = g.get_num_nodes();
    time_counter _t1, _t2, _t3;
    for (size_t i = 0; i < queries.size(); i += 2) {
        node u = queries[i] % n;
        node v = queries[i + 1] % n;
        if (!origin_to_new.empty()) {
            u = origin_to_new[u];
            v = origin_to_new[v];
        }

        _t1.start();
        has_edge_cnt_csr += g.has_edge(u, v);
        _t1.stop();

        if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
            // printf("u: %d, v: %d\n", u, v);
            _t2.start();
            has_edge_cnt_comp += g_comp.has_directed_edge(u, v);
            _t2.stop();
        }

        _t3.start();
        has_edge_cnt_hvi += g_hvi.has_edge(u, v);
        _t3.stop();
    }
    _t1.print("Edge existance test with CSR format");
    printf("(CSR) Has edge cnt: %lu.\n", has_edge_cnt_csr);
    if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
        _t2.print("Edge existance test with CompressGraph");
        printf("(CompressGraph) Has edge cnt: %lu.\n", has_edge_cnt_comp);
    }
    _t3.print("Edge existance test with HVI");
    printf("(HVI) Has edge cnt: %lu.\n", has_edge_cnt_hvi);
}

void bfs_test(string querypath, Graph &g, CompressGraph &g_comp, HVI &g_hvi, vector<node> &origin_to_new) {
    vector<node> queries;
    read_node_queries(querypath, queries);
    size_t test_pass_cnt_comp = 0, test_pass_cnt_hvi = 0, n = g.get_num_nodes();
    vector<size_t> bfs_rank1;
    vector<size_t> bfs_rank2;
    vector<size_t> bfs_rank3;
    time_counter _t1, _t2, _t3;
    for (node v: queries) {
        v = v % n;
        if (!origin_to_new.empty()) v = origin_to_new[v];

        _t1.start();
        g.bfs(v, bfs_rank1);
        _t1.stop();

        if (origin_to_new.empty() && g_comp.get_num_rules() != 0){
            _t2.start();
            g_comp.bfs(v, bfs_rank2);
            _t2.stop();
        }
        
        _t3.start();
        g_hvi.bfs(v, bfs_rank3);
        _t3.stop();

        if (origin_to_new.empty() && g_comp.get_num_rules() != 0)
            if (vec_fast_verification(bfs_rank1, bfs_rank2)) test_pass_cnt_comp++;
        if (vec_fast_verification(bfs_rank1, bfs_rank3)) test_pass_cnt_hvi++;
    }
    _t1.print("bfs with CSR format");
    if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
        _t2.print("bfs with CompressGraph");
        printf("(CompressGraph) Pass test cnt: %lu.\n", test_pass_cnt_comp);
    }
    _t3.print("bfs with HVI");
    printf("(HVI) Pass test cnt: %lu.\n", test_pass_cnt_hvi);
}

void cc_test(Graph &g, CompressGraph &g_comp, HVI &g_hvi, vector<node> &origin_to_new) {
    vector<node> component_id_1;
    vector<node> component_id_2;
    vector<node> component_id_3;
    time_counter _t1, _t2, _t3;
    const size_t n = g.get_num_nodes();

    _t1.start();
    g.cc(component_id_1);
    _t1.stop();
    _t1.print("computing connected components with CSR format");
    print_vec<node>(component_id_1, 10);

    if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
        printf("#rules of CompressGraph: %lu.\n", g_comp.get_num_rules());
        _t2.start();
        g_comp.cc(component_id_2);
        _t2.stop();
        _t2.print("computing connected components with CompressGraph");
        print_vec<node>(component_id_2, 10);
        component_id_2.erase(component_id_2.begin() + n, component_id_2.end());
        if (vec_fast_verification(component_id_1, component_id_2)) printf("(CompressGraph) Connect component test: Pass.\n"); 
        else printf("(CompressGraph) Connect component test: Error.\n");
    }
    
    _t3.start();
    g_hvi.cc(component_id_3);
    _t3.stop();
    _t3.print("computing connected components with HVI");
    print_vec<node>(component_id_3, 10);
    if (vec_fast_verification(component_id_1, component_id_3)) printf("(HVI) Connect component test: Pass.\n"); 
    else printf("(HVI) Connect component test: Error.\n");
}

void core_decomposition_test(Graph &g, CompressGraph &g_comp, HVI &g_hvi, vector<node> origin_to_new) {
    vector<size_t> core_numbers_1;
    vector<size_t> core_numbers_2;
    vector<size_t> core_numbers_3;
    time_counter _t1, _t2, _t3;

    _t1.start();
    g.core_decomposition(core_numbers_1);
    _t1.stop();
    _t1.print("computing core numbers with CSR format");

    if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
        _t2.start();
        g_comp.core_decomposition(core_numbers_2);
        _t2.stop();
        _t2.print("computing core numbers with CompressGraph");
        if (vec_fast_verification(core_numbers_1, core_numbers_2)) printf("(CompressGraph) Core decomposition test: Pass.\n"); 
        else printf("(CompressGraph) Core decomposition test: Error.\n");
    }

    _t3.start();
    g_hvi.core_decomposition(core_numbers_3);
    _t3.stop();
    _t3.print("computing core numbers with HVI");
    if (vec_fast_verification(core_numbers_1, core_numbers_3)) printf("(HVI) Core decomposition test: Pass.\n"); 
    else printf("(HVI) Core decomposition test: Error.\n");
}

void sm_test(Graph &g, CompressGraph &g_comp, HVI &g_hvi, string querypath) {
    Graph q(querypath + "csr_vlist.bin", querypath + "csr_elist.bin");
    time_counter _t1, _t2, _t3;
    ull res;
    _t1.start();
    res = g.subgraph_matching(q);
    _t1.stop();
    _t1.print("subgraph matching with CSR format");
    printf("Number of matchings: %llu.\n", res);

    _t2.start();
    res = g_hvi.subgraph_matching(q);
    _t2.stop();
    _t2.print("subgraph matching with HVI");
    printf("Number of matchings: %llu.\n", res);

    // _t3.start();
    // res = g_comp.subgraph_matching(q);
    // _t3.stop();
    // _t3.print("subgraph matching with CompressGraph");
    // printf("Number of matchings: %llu.\n", res);
}

void tc_test(Graph &g, CompressGraph &g_comp, HVI &g_hvi, vector<node> origin_to_new) {
    time_counter _t1, _t2, _t3;
    _t1.start();
    size_t res = g.tc();
    _t1.stop();
    _t1.print("TC with CSR format");
    printf("Number of triangles: %lu.\n", res);

    if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
         _t2.start();
        res = g_comp.tc();
        _t2.stop();
        _t2.print("TC with CompressGraph");
        printf("Number of triangles: %lu.\n", res);
    }
   
    _t3.start();
    res = g_hvi.tc();
    _t3.stop();
    _t3.print("TC with HVI");
    printf("Number of triangles: %lu.\n", res);
}

void page_rank_test(Graph &g, CompressGraph &g_comp, HVI &g_hvi, vector<node> origin_to_new) {
    const int max_iter = 100;
    time_counter _t1, _t2, _t3;
    vector<double> pr_val_1, pr_val_2, pr_val_3;
    _t1.start();
    g.page_rank(pr_val_1, 0.05, max_iter);
    _t1.stop();
    _t1.print("computing PageRank with CSR format");

    if (origin_to_new.empty() && g_comp.get_num_rules() != 0) {
        _t2.start();
        g_comp.page_rank(pr_val_2, 0.05, max_iter);
        _t2.stop();
        _t2.print("computing PageRank with CompressGraph");
    }

    _t3.start();
    g_hvi.page_rank(pr_val_3, 0.05, max_iter);
    _t3.stop();
    _t3.print("computing PageRank with HVI");
}

int main(int argc, char **argv) {
    string taskname = argv[1];
    if (taskname == "genq") {
        string query_class = argv[2];
        if (query_class == "v") generate_node_queries(stoul(argv[3]), argv[4]);
        if (query_class == "e") generate_edge_queries(stoul(argv[3]), argv[4]);
        if (query_class == "sg") for (int i = 0; i < 6; i++) generate_subgraph_query(i, argv[3]);
    }
    else {
        string graphdir = argv[2];
        string orderingmethod = argv[3];
        time_counter _t_g, _t_comp, _t_hvi;
        _t_g.start();
        Graph g(graphdir + "/" + orderingmethod + "/csr_vlist.bin", graphdir + "/" + orderingmethod + "/csr_elist.bin");
        _t_g.stop();
        _t_g.print("reading graph from binfile");
        size_t _size = g.size_in_bytes();
        printf("Size of graph: %.3f MB.\n", (float) _size / (1 << 20));
        size_t num_nodes = g.get_num_nodes();
        _t_comp.start();
        CompressGraph g_comp(graphdir + "/compress/csr_vlist.bin", graphdir + "/compress/csr_elist.bin", num_nodes);
        _t_comp.stop();
        _t_comp.print("reading CompressGraph from binfile");
        _size = g_comp.size_in_bytes();
        printf("Size of CompressGraph: %.3f MB.\n", (float) _size / (1 << 20));
        _t_hvi.start();
        // HVI g_hvi(g);
        HVI g_hvi(graphdir + "/" + orderingmethod + "/hvi_offsets.bin", graphdir + "/" + orderingmethod + "/hvi_list.bin");
        _t_hvi.stop();
        _t_hvi.print("reading HVI from binfile");
        _size = g_hvi.size_in_bytes();
        printf("Size of HVI: %.3f MB.\n", (float) _size / (1 << 20));
        vector<node> origin_to_new;
        if (orderingmethod != "origin") read_ordering(graphdir + "/" + orderingmethod + "/origin2new.bin", origin_to_new);
        if (taskname == "ee") ee_test(argv[4], g, g_comp, g_hvi, origin_to_new);
        if (taskname == "bfs") bfs_test(argv[4], g, g_comp, g_hvi, origin_to_new);
        if (taskname == "cc") cc_test(g, g_comp, g_hvi, origin_to_new);
        if (taskname == "kcore") core_decomposition_test(g, g_comp, g_hvi, origin_to_new);
        if (taskname == "sm") sm_test(g, g_comp, g_hvi, argv[4]);
        if (taskname == "tc") tc_test(g, g_comp, g_hvi, origin_to_new);
        if (taskname == "pr") page_rank_test(g, g_comp, g_hvi, origin_to_new);
        if (taskname == "neis") neis_test(g, g_comp);
    }
    
    // neis_test(graphdir);
}