#include "Graph.hpp"
#include "Interval.hpp"

string get_graphdict(string graphname) {
    for (auto k : files) {
        if (graphname != k.first) continue;
        printf("==============Graph %s ==================\n", k.first.c_str());
        return dict_path + k.second;
    }
}

string get_outputdict(string graphname, string orderingmethod) {
    if (!ordering_methods.count(orderingmethod)) {
        printf("Invalid ordering method! Please select from Greedy, MWPM, and Serdyukov.\n");
        return "";
    }
    for (auto k : files) {
        if (graphname != k.first) continue; 
        return ordering_root + orderingmethod + "/" + graphname + "_newID.txt";
    }
}

void generate_and_save_node_queries(size_t num_node_queries, string outputdict) {
    ofstream f(outputdict);
    srand(chrono::system_clock::now().time_since_epoch().count());
    for (size_t i = 0; i < num_node_queries; ++i) {
        f << rand() << endl;
    }
    f.close(); 
}

void generate_and_save_edge_queries(size_t num_edge_pairs, string outputdict) {
    ofstream f(outputdict);
    for (size_t i = 0; i < num_edge_pairs; ++i) {
        f << rand() << "\t" << rand() << endl;
    }
    f.close();
}

void read_edge_queries(size_t num_nodes, string inputdict, vector<pair<node, node>> &edge_queries) {
    ifstream f(inputdict);
    string line;
    node u, v;
    while (getline(f, line)){
        stringstream ss(line);
        ss >> u >> v;
        edge_queries.push_back({u % num_nodes ,v % num_nodes});
    }
    f.close();
}

void read_node_queries(size_t num_nodes, string inputdict, vector<node> &node_queries) {
    ifstream f(inputdict);
    string line;
    node u;
    while (getline(f, line)){
        stringstream ss(line);
        ss >> u;
        node_queries.push_back(u % num_nodes);
    }
    f.close(); 
}

void save_ordering(string outputdir, vector<node> &origin_to_new) {
    ofstream out(outputdir);
	for (int i = 0; i < origin_to_new.size(); i++) {
		out << i << " " << origin_to_new[i] << endl;
	}
	out.close();
}

void reorder(string graphname, string orderingmethod) {
    if (!ordering_methods.count(orderingmethod)) {
        printf("Invalid ordering method! Please select from Greedy, MWPM, and Serdyukov.\n");
        return;
    }
    string graphdict = get_graphdict(graphname);
    string outputdict = get_outputdict(graphname, orderingmethod);
    time_counter _t;
    _t.start();
    Graph G(graphdict);
    _t.stop();
    printf("Has read graph from %s.\n", graphdict.c_str());
    _t.print("graph construction");  

    size_t num_nodes = G.get_num_nodes();
    size_t deg_upper_bound = (size_t)(G.get_num_edges() / num_nodes) * 5;
    vector<vector<pair<node, size_t>>> edges;
    edges.resize(num_nodes); 

    size_t _per = num_nodes / 100, _per_cnt = 0;
    srand(chrono::system_clock::now().time_since_epoch().count());
    vector<node> rands;
    rands.reserve(2000);
    for (size_t i = 0; i < 2000; i++) {
        rands.push_back(rand());
    }
    _t.clear();
    _t.start();
    int num_threads = 10;
    omp_set_num_threads(num_threads);
    vector<vector<node>> candidates(num_threads, vector<node>(1000, UINT_MAX));
    vector<vector<bool>> seen(num_threads, vector<bool>(num_nodes, false));
    vector<vector<pair<node, weight>>> res(num_threads, vector<pair<node, weight>>(1000, {UINT_MAX, UINT_MAX}));
    #pragma omp parallel 
    {
        for (node i = 0; i < num_nodes; i++) {
            int _tid = omp_get_thread_num();
            if (i % num_threads != _tid) continue;
            fill(seen[_tid].begin(), seen[_tid].end(), false);
            seen[_tid][i] = true;
            candidates[_tid].clear();
            size_t num_samples = min((size_t) 1000, 5 * G.get_number_of_neighbors(i));
            size_t _res_cnt = 0, _can_cnt = 0;
            for (size_t j = 0; j < num_samples; j++) {
                node v = G.get_one_neighbor(i, rands[(i+j*2) % 2000] % G.get_number_of_neighbors(i));
                node w = G.get_one_neighbor(v, rands[(i+2*j+1) % 2000] % G.get_number_of_neighbors(v));
                if (!seen[_tid][w]) {
                    candidates[_tid][_can_cnt++] = w;
                    seen[_tid][w] = true;
                }
            }
            if (_can_cnt == 0) continue;
            for (size_t k = 0; k < _can_cnt; k++) {
                res[_tid][_res_cnt++] = {candidates[_tid][k], G.cnt_common_neighbors(i, candidates[_tid][k])};
            }
            sort(res[_tid].begin(), res[_tid].begin() + min(deg_upper_bound, _res_cnt), weight_greater);
            edges[i].insert(edges[i].begin(), res[_tid].begin(), res[_tid].begin() + min(deg_upper_bound, _res_cnt));
        }
    }
    WeightedGraph WG(edges);
    _t.stop();
    _t.print("simplified graph construction");
    vector<vector<node>>().swap(candidates);
    vector<vector<bool>>().swap(seen);
    vector<vector<pair<node, weight>>>().swap(res);
    vector<vector<pair<node, size_t>>>().swap(edges);

    vector<node> origin_to_new;
    if (orderingmethod == "Greedy") {
        WG.get_ordering_greedy(origin_to_new);
    }
    if (orderingmethod == "MWPM") {
        vector<pair<node, node>> PM(num_nodes, {UINT_MAX, UINT_MAX});
        WG.maximum_perfect_matching_greedy(PM);
        WG.expand_perfect_matching_to_ordering(PM, origin_to_new);
    }
    if (orderingmethod == "Serdyukov") {
        vector<pair<node, node>> CC(num_nodes, {UINT_MAX, UINT_MAX});
        WG.maximun_cycle_cover_greedy(CC);
        WG.expand_cycle_cover_to_ordering(CC, origin_to_new);
    }
    save_ordering(outputdict, origin_to_new);
    G.reorder(origin_to_new);
    HVI g_hvi(G);
    ull _size = g_hvi.size_in_bytes();
    printf("Size of HVI with new ordering: %.3f MB!\n", (float)_size / (1 << 20));
}

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

void bfs_test(string graphdict, string inputdict) {
    size_t bfs_verification_pass_cnt = 0;
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    HVI g_hvi(g);
    ull _size = g_hvi.size_in_bytes();
    printf("Size of HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    int num_nodes = g.get_num_nodes();
    vector<int> bfs_rank1;
    vector<int> gt_bfs_rank;
    vector<node> node_queries;
    bfs_rank1.resize(num_nodes);
    gt_bfs_rank.resize(num_nodes);
    read_node_queries(num_nodes, inputdict, node_queries);
    int cnt = 0;

    for (auto u : node_queries) {
        fill(gt_bfs_rank.begin(), gt_bfs_rank.end(), INT_MAX);
        _t_g.start();
        g.bfs(u, gt_bfs_rank);
        _t_g.stop();
        
        fill(bfs_rank1.begin(), bfs_rank1.end(), INT_MAX);
        _t_hvi.start();
        g_hvi.bfs(u, bfs_rank1);
        _t_hvi.stop();
    
        if (bfs_fast_verification(bfs_rank1, gt_bfs_rank)) bfs_verification_pass_cnt++;
        printf("\r[%d/100] BFS queries have been done!", ++cnt);
        fflush(stdout);
    }
    printf("\n");
    _t_g.print("BFS with adjacency list");
    _t_hvi.print("BFS with HVI");
    printf("#BFS verification pass: %lu.\n", bfs_verification_pass_cnt);
}

void bfs_test(string graphdict, string orderingdict, string inputdict) {
    size_t bfs_verification_pass_cnt = 0;
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    int num_nodes = g.get_num_nodes();
    vector<int> bfs_rank;
    vector<int> gt_bfs_rank;
    vector<node> node_queries;
    vector<node> origin_to_new(num_nodes, 0);
    bfs_rank.resize(num_nodes);
    gt_bfs_rank.resize(num_nodes);
    printf("%s\n",orderingdict.c_str());
    read_ordering(orderingdict, origin_to_new, num_nodes);
    read_node_queries(num_nodes, inputdict, node_queries);
    g.reorder(origin_to_new);
    for (size_t i = 0; i < node_queries.size(); ++i) {
        node_queries[i] = origin_to_new[node_queries[i]];
    }
    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));
    int cnt = 0;

    for (auto u : node_queries) {
        fill(gt_bfs_rank.begin(), gt_bfs_rank.end(), INT_MAX);
        _t_g.start();
        g.bfs(u, gt_bfs_rank);
        _t_g.stop();

        fill(bfs_rank.begin(), bfs_rank.end(), INT_MAX);
        _t_hvi.start();
        g_hvi.bfs(u, bfs_rank);
        _t_hvi.stop();

        if (bfs_fast_verification(bfs_rank, gt_bfs_rank)) bfs_verification_pass_cnt++;
        printf("\r[%d/100] BFS queries have been done!", ++cnt);
        fflush(stdout);
    }
    printf("\n");
    _t_g.print("BFS with adjacency list");
    _t_hvi.print("BFS with HVI");
    printf("#Bfs verification pass: %lu.\n", bfs_verification_pass_cnt);
}

void tc(string graphdict) {
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    _t_g.start();
    ull res1 = g.cnt_tri_merge();
    _t_g.stop();
    _t_g.print("tringle counting with adjacency list");
    printf("Number of triangles: %llu!\n", res1);

    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    size_t res2 = g_hvi.cnt_tri_merge();
    _t_hvi.stop();
    _t_hvi.print("tringle counting with HVI");
    printf("Number of triangles: %lu!\n", res2);
}

void tc(string graphdict, string orderingdict) {
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    int num_nodes = g.get_num_nodes();
    vector<node> origin_to_new(num_nodes, 0);
    printf("%s\n",orderingdict.c_str());
    read_ordering(orderingdict, origin_to_new, num_nodes);
    g.reorder(origin_to_new);

    _t_g.start();
    ull res1 = g.cnt_tri_merge();
    _t_g.stop();
    _t_g.print("tringle counting with adjacency list");
    printf("Number of triangles: %llu!\n", res1);

    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    size_t res2 = g_hvi.cnt_tri_merge();
    _t_hvi.stop();
    _t_hvi.print("tringle counting with HVI");
    printf("Number of triangles: %lu!\n", res2);
}

void has_edge_test(string graphdict, string edgequerydict) {
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    size_t num_nodes = g.get_num_nodes();
    vector<pair<node, node>> edge_queries;
    read_edge_queries(num_nodes, edgequerydict, edge_queries);

    _t_g.start();
    size_t res1 = 0;
    for (auto q : edge_queries) {
        if (g.has_undirected_edge(q.first, q.second)) res1++;
    }
    _t_g.stop();
    _t_g.print("edge existance test with adjacency list");
    printf("Number of exist edges: %lu!\n", res1);

    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    size_t res2 = 0;
    for (auto q : edge_queries) {
        if (g_hvi.has_edge(q.first, q.second)) res2++;
    }
    _t_hvi.stop();
    _t_hvi.print("edge existance test with HVI");
    printf("Number of exist edges: %lu!\n", res2);
}

void has_edge_test(string graphdict, string edgequerydict, string orderingdict) {
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    int num_nodes = g.get_num_nodes();
    vector<node> origin_to_new(num_nodes, 0);
    printf("%s\n",orderingdict.c_str());
    read_ordering(orderingdict, origin_to_new, num_nodes);
    g.reorder(origin_to_new);

    vector<pair<node, node>> edge_queries;
    read_edge_queries(num_nodes, edgequerydict, edge_queries);
    for (size_t i = 0; i < edge_queries.size(); ++i) {
        edge_queries[i] = {origin_to_new[edge_queries[i].first], origin_to_new[edge_queries[i].second]};
    }

    _t_g.start();
    ull res1 = 0;
    for (auto q : edge_queries) {
        if (g.has_undirected_edge(q.first, q.second)) res1++;
    }
    _t_g.stop();
    _t_g.print("edge existance test with adjacency list");
    printf("Number of exist edges: %llu!\n", res1);

    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    ull res2 = 0;
    for (auto q : edge_queries) {
        if (g_hvi.has_edge(q.first, q.second)) res2++;
    }
    _t_hvi.stop();
    _t_hvi.print("edge existance test with HVI");
    printf("Number of exist edges: %llu!\n", res2);
}

void page_rank(string graphdict) {
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");

    vector<double> pr;
    vector<double> pr_hvi;
    _t_g.start();
    g.page_rank(pr, 0.01, 100);
    _t_g.stop();
    _t_g.print("PageRank computation with adjacency list");

    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    g_hvi.page_rank(pr_hvi, 0.01, 100);
    _t_hvi.stop();
    _t_hvi.print("PageRank computation with HVI");
}

void page_rank(string graphdict, string orderingdict) {
    time_counter _t, _t_g, _t_hvi;
    _t.start();
    Graph g(graphdict);
    _t.stop();
    _t.print("graph construction");
    int num_nodes = g.get_num_nodes();
    vector<node> origin_to_new(num_nodes, 0);
    printf("%s\n",orderingdict.c_str());
    read_ordering(orderingdict, origin_to_new, num_nodes);
    g.reorder(origin_to_new);

    vector<double> pr;
    vector<double> pr_hvi;
    _t_g.start();
    g.page_rank(pr, 0.01, 100);
    _t_g.stop();
    _t_g.print("PageRank computation with adjacency list");

    HVI g_hvi(g);
    size_t _size = g_hvi.size_in_bytes();
    printf("Size of new HVI representation: %.3f MB!\n", (float)_size / (1 << 20));

    _t_hvi.start();
    g_hvi.page_rank(pr_hvi, 0.01, 100);
    _t_hvi.stop();
    _t_hvi.print("PageRank computation with HVI");
}

int main(int argc, char **argv) {
    string task = argv[1];
    if (task == "genq") {
        if (argv[2] == "e") generate_and_save_edge_queries(stoi(argv[3]), "./edge_queries.txt");
        if ((string)argv[2] == "v") generate_and_save_node_queries(stoi(argv[3]), "./node_queries.txt");
    }
    if (task == "reorder") {
	    reorder(argv[2], argv[3]);
    }

    if (task == "bfs") {
        string graphdict = get_graphdict(argv[2]);
        string nodequerydict = "./node_queries.txt";
	    if (argc == 3) bfs_test(graphdict, nodequerydict);
        else if (argc == 4) {
            string orderingdict = get_outputdict(argv[2], argv[3]);
            bfs_test(graphdict, orderingdict, nodequerydict);
        }
    }

    if (task == "tc") {
        string graphdict = get_graphdict(argv[2]);
	    if (argc == 3) tc(graphdict);
        else if (argc == 4) {
            string orderingdict = get_outputdict(argv[2], argv[3]);
            tc(graphdict, orderingdict);
        }
    }

    if (task == "he") {
        string graphdict = get_graphdict(argv[2]);
        string edgequerydict = "./edge_queries.txt";
	    if (argc == 3) has_edge_test(graphdict, edgequerydict);
        else if (argc == 4) {
            string orderingdict = get_outputdict(argv[2], argv[3]);
            has_edge_test(graphdict, edgequerydict, orderingdict);
        }
    }

    if (task == "pr") {
        string graphdict = get_graphdict(argv[2]);
	    if (argc == 3) page_rank(graphdict);
        else if (argc == 4) {
            string orderingdict = get_outputdict(argv[2], argv[3]);
            page_rank(graphdict, orderingdict);
        }
    }
    return 0;
}