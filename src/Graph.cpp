#include "Graph.hpp"

using namespace std;

void time_counter::start() {
    t_start = chrono::steady_clock::now();
}

void time_counter::stop() {
    t_end = chrono::steady_clock::now();
    t_cnt += chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count();
}

void time_counter::print(string s) {
    if (t_cnt < 1000) printf("Time used for %s: %llu ns.\n", s.c_str(), t_cnt);
    else if (t_cnt < 1000000) printf("Time used for %s: %.3f us.\n", s.c_str(), (float) t_cnt / 1000);
    else if (t_cnt < (ull) 1000000000) printf("Time used for %s: %.3f ms.\n", s.c_str(), (float) t_cnt / 1000000);
    else printf("Time used for %s: %.3f s.\n", s.c_str(), (float) t_cnt / (float) 1000000000);
}

void time_counter::clean() {
    t_cnt = 0;
}

template <class T>
bool weight_less(const std::pair<node, T> &x, const std::pair<node, T> &y) {
    return x.second < y.second;
}
template bool weight_less(const std::pair<node, int> &x, const std::pair<node, int> &y);

template <class T>
bool weight_greater(const std::pair<node, T> &x, const std::pair<node, T> &y) {
    return x.second > y.second;
}
template bool weight_greater(const std::pair<node, int> &x, const std::pair<node, int> &y);

template <class T>
bool vid_less(const std::pair<node, T> &x, const std::pair<node, T> &y) {
    return x.first < y.first;
}
template bool vid_less(const std::pair<node, int> &x, const std::pair<node, int> &y);

template <class T>
bool vid_greater(const std::pair<node, T> &x, const std::pair<node, T> &y) {
    return x.first > y.first;
}
template bool vid_greater(const std::pair<node, int> &x, const std::pair<node, int> &y);

Graph::Graph(const string graphfile) {
    bool has_zero = false;
	num_nodes = 0;
	num_edges = 0;
	FILE *fp = fopen(graphfile.c_str(), "r");
	if (fp == NULL) {
        std::cout << "fail to open " << graphfile << std::endl;
    }
	vector<pair<node, node>> edge_list;

    char line[512];
    while (fgets(line, 512, fp) != NULL) {
        if (line[0] == '#' || line[0] == '%') continue;
        int u = 0, v = 0;
        const char *c = line;
        while (isdigit(*c))
            u = (u << 1) + (u << 3) + (*c++ - 48);
        c++;
        while (isdigit(*c))
            v = (v << 1) + (v << 3) + (*c++ - 48);
		if (u == 0 || v == 0) {
			has_zero = true;
		}
		if (u == v) continue;
		if (u > num_nodes) num_nodes = u;
		if (v > num_nodes) num_nodes = v;
        edge_list.push_back({(node) u, (node) v});
    }    
    fclose(fp);

	if (has_zero) num_nodes++;
    vector<vector<node>> adj_list(num_nodes);  
	for (auto e : edge_list) {
		if (has_zero) {
			adj_list[e.first].push_back(e.second);
			adj_list[e.second].push_back(e.first);
		}
		else {
			adj_list[e.first-1].push_back(e.second-1);
			adj_list[e.second-1].push_back(e.first-1);
		}
	}
    vector<pair<node, node>>().swap(edge_list);
	for (int i = 0; i < num_nodes; ++i){
		if (!adj_list[i].empty()) {
			sort(adj_list[i].begin(), adj_list[i].end());
			vector<node> tmp_neis(adj_list[i]);
			adj_list[i].clear();
			node prev = INT_MAX;
			for (auto j : tmp_neis) {
				if (j == prev) continue;
				prev = j;
				adj_list[i].push_back(j);
			}
			num_edges += adj_list[i].size();
		} 
	}
	
    csr_vlist.reserve(num_nodes + 1);
    csr_elist.reserve(num_edges);
    for (size_t i = 0; i < num_nodes; i++) {
        csr_vlist.push_back(csr_elist.size());
        csr_elist.insert(csr_elist.end(), adj_list[i].begin(), adj_list[i].end());
    }
    csr_vlist.push_back(num_edges);
    vector<vector<node>>().swap(adj_list);

	printf("Number of nodes: %lu, number of edges: %lu, size of adj-list: %.3f MB!\n", 
			num_nodes, num_edges, (float)(num_edges + num_nodes) * sizeof(node) / (1<<20));
}

Graph::Graph(const string csr_vlist_binfile, const string csr_elist_binfile) {
    ifstream input_csr_vlist(csr_vlist_binfile, ios::binary);
    if (!input_csr_vlist.is_open()) {
        cerr << "Unable to open csr vlist binfile from " << csr_vlist_binfile << endl;
        return;
    }
    input_csr_vlist.seekg(0, ios::end);
    streamsize size = input_csr_vlist.tellg();
    input_csr_vlist.seekg(0, ios::beg);
    csr_vlist.resize(size / sizeof(int));
    input_csr_vlist.read(reinterpret_cast<char*>(csr_vlist.data()), size);
    input_csr_vlist.close();
    num_nodes = csr_vlist.size() - 1;

    ifstream input_csr_elist(csr_elist_binfile, ios::binary);
    if (!input_csr_elist.is_open()) {
        cerr << "Unable to open csr elist binfile from " << csr_elist_binfile << endl;
        return;
    }
    input_csr_elist.seekg(0, ios::end);
    size = input_csr_elist.tellg();
    input_csr_elist.seekg(0, ios::beg);
    csr_elist.resize(size / sizeof(node));
    input_csr_elist.read(reinterpret_cast<char*>(csr_elist.data()), size);
    input_csr_elist.close();
    num_edges = csr_elist.size();
    printf("Has read graph! #Vertices: %lu, #Edges: %lu.\n", num_nodes, num_edges);
}

Graph::Graph(const vector<vector<node>> &adj_list) {
    num_nodes = adj_list.size();
    csr_vlist.reserve(num_nodes + 1);
    csr_vlist.push_back(0);
    for (auto neis: adj_list) {
        csr_elist.insert(csr_elist.end(), neis.begin(), neis.end());
        csr_vlist.push_back(csr_elist.size());
    }
    num_edges = csr_elist.size();
    printf("Has constructed graph! #Vertices: %lu, #Edges: %lu.\n", num_nodes, num_edges);
}

const int* Graph::get_csr_vlist() const {
    return csr_vlist.data();
}

const node* Graph::get_csr_elist() const {
    return csr_elist.data();
}

bool Graph::has_edge(node u, node v) const {
    return get_deg(u) < get_deg(v) ? has_directed_edge(u, v) : has_directed_edge(v, u);
}

bool Graph::has_directed_edge(node u, node v) const {
    size_t ptr = lower_bound(csr_elist.begin() + csr_vlist[u], csr_elist.begin() + csr_vlist[u+1], v) - csr_elist.begin();
    return ptr < csr_vlist[u+1] && csr_elist[ptr] == v;
}

node Graph::get_nei(node u, size_t offset) const {
    return csr_elist[csr_vlist[u] + offset];
}

size_t Graph::cnt_common_neis(node u, node v) const {
    size_t u_ptr = csr_vlist[u], v_ptr = csr_vlist[v], res = 0;
    while (u_ptr < csr_vlist[u+1] && v_ptr < csr_vlist[v+1]) {
        if (csr_elist[u_ptr] == csr_elist[v_ptr]) {
            res++;
            u_ptr++;
            v_ptr++;
        }
        else if (csr_elist[u_ptr] < csr_elist[v_ptr]) u_ptr++;
        else v_ptr++;
    }
    return res;
}

size_t Graph::cnt_common_neis(node u, const vector<node> &nodes) const {
    size_t u_ptr = csr_vlist[u], v_ptr = 0, res = 0;
    while (u_ptr < csr_vlist[u+1] && v_ptr < nodes.size()) {
        if (csr_elist[u_ptr] == nodes[v_ptr]) {
            res++;
            u_ptr++;
            v_ptr++;
        }
        else if (csr_elist[u_ptr] < nodes[v_ptr]) u_ptr++;
        else v_ptr++;
    }
    return res;
}

size_t Graph::get_num_edges() const {
    return num_edges;
}

size_t Graph::get_num_nodes() const {
    return num_nodes;
}

size_t Graph::get_deg(node u) const {
    return csr_vlist[u+1] - csr_vlist[u];
}

ull Graph::size_in_bytes() const {
    return csr_vlist.size() * sizeof(int) + csr_elist.size() * sizeof(node);
}

void Graph::get_common_neis(node u, node v, vector<node>& res) const {
	int u_ind = csr_vlist[u], v_ind = csr_vlist[v];
	while(u_ind < csr_vlist[u+1] && v_ind < csr_vlist[v+1]) {
		if (csr_elist[u_ind] == csr_elist[v_ind]) {
			res.emplace_back(csr_elist[u_ind]);
			u_ind++;
			v_ind++;
		}
		else if (csr_elist[u_ind] < csr_elist[v_ind]) u_ind++;
		else v_ind++;
	}
}

void Graph::get_common_neis(node u, const vector<node> &nodes, vector<node> &res) const {
	res.clear();
	int u_ind = csr_vlist[u], v_ind = 0;
	while (u_ind < csr_vlist[u+1] && v_ind < nodes.size()) {
		if (csr_elist[u_ind] == nodes[v_ind]) {
			res.push_back(nodes[v_ind]);
			u_ind++;
			v_ind++;
		}
		else if (csr_elist[u_ind] < nodes[v_ind]) u_ind++;
		else v_ind++;
	}
}

void Graph::get_degs(vector<int> &degs) const {
    degs.clear();
    degs.resize(num_nodes);
    for (node u = 0; u < num_nodes; u++) {
        degs[u] = csr_vlist[u+1] - csr_vlist[u];
    }
}

void Graph::get_neis(node u, vector<node> &neis) const {
    neis.clear();
    if (csr_vlist[u] == csr_vlist[u+1]) return;
    neis.reserve(csr_vlist[u+1] - csr_vlist[u]);
    neis.insert(neis.end(), csr_elist.begin() + csr_vlist[u], csr_elist.begin() + csr_vlist[u+1]);
}

void Graph::print_neis(node u) const {
    printf("N(%d):", u);
    for (int i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
        printf(" %d", csr_elist[i]);
    }
    printf(".\n");
}

void Graph::reorder(const vector<node> &origin_to_new){
	vector<bool> seen(num_nodes, false);
	vector<vector<node>> adj_list(num_nodes);
	for(size_t i = 0; i < num_nodes; ++i) {
		node new_id = origin_to_new[i];
		if (new_id == UINT_MAX) continue;
		if (seen[new_id]) printf("Error on node %lu! Already has node with label %d!\n", i, new_id);
		else seen[new_id] = true;
        for (size_t j = csr_vlist[i]; j < csr_vlist[i+1]; ++j) {
            node v = csr_elist[j];
            adj_list[new_id].push_back(origin_to_new[v]);
        }
		if (!adj_list[new_id].empty()) sort(adj_list[new_id].begin(), adj_list[new_id].end());
	}
    csr_vlist.clear();
    csr_elist.clear();
    for (size_t i = 0; i < num_nodes; i++) {
        csr_vlist.push_back(csr_elist.size());
        csr_elist.insert(csr_elist.end(), adj_list[i].begin(), adj_list[i].end());
    }
    csr_vlist.push_back(csr_elist.size());
}

void Graph::save_as_csr_binfile(const string csr_vlist_binfile, const string csr_elist_binfile) {
    ofstream output_csr_vlist_binfile(csr_vlist_binfile, ios::binary);
    output_csr_vlist_binfile.write(reinterpret_cast<const char*>(csr_vlist.data()), sizeof(int) * csr_vlist.size());
    printf("Size of csr_vlist: %lu.\n", csr_vlist.size());
    output_csr_vlist_binfile.close();

    ofstream output_csr_elist_binfile(csr_elist_binfile, ios::binary);
    output_csr_elist_binfile.write(reinterpret_cast<const char*>(csr_elist.data()), sizeof(node) * csr_elist.size());
    printf("Size of csr_elist: %lu.\n", csr_elist.size());
    output_csr_elist_binfile.close();
}

size_t Graph::tc() {
    size_t res = 0;
    for (node u = 0; u < num_nodes; u++) {
        for (size_t i = csr_vlist[u]; i < csr_vlist[u + 1]; i++) {
            node v = csr_elist[i];
            res += cnt_common_neis(u, v);
        }
    }
    return res;
}

ull Graph::mc(node u) {
	ull res = 0;
	vector<node> P;
	get_neis(u, P);
	vector<node> X;
	X.reserve(csr_vlist[u+1] - csr_vlist[u]);
	BKP(P, X, res);
	return res;
}

ull Graph::subgraph_matching(const Graph &q) {
    ull res = 0;
    vector<node> join_order;
    size_t n = q.get_num_nodes();
    join_order.reserve(n);
    node first_join_node = 0;
    size_t q_max_deg = q.get_deg(0);
    for (node i = 1; i < n; i++) {
        if (q.get_deg(i) > q_max_deg) {
            q_max_deg = q.get_deg(i);
            first_join_node = i;
        }
    }

    q.get_dfs_order(first_join_node, join_order);
    if (join_order.size() < n) {
        printf("query graph is not connected!\n");
        return res;
    }

    // vector<vector<node>> inter_res(n);
    // unordered_set<node> partial_matching;
    vector<node> partial_matching;
    partial_matching.reserve(n);
    int _per_cnt = (num_nodes - 1) / 100 + 1, _cnt = 0;
    for (node u = 0; u < num_nodes; u++) {
        if (u % _per_cnt == 0){
            printf("\rMatching Process: [%d/100].", ++_cnt);
            fflush(stdout);
        }
        if (get_deg(u) < q_max_deg) continue;
        // inter_res[0].push_back(u);
        partial_matching.push_back(u);
        // partial_matching.insert(u);
        subgraph_matching_helper(q, partial_matching, join_order, res);
        // inter_res[0].pop_back();
        partial_matching.pop_back();
        partial_matching.clear();
    }
    printf("\rMatching Process: [100/100].\n");
    return res;
}

void Graph::BKP(vector<node> &P, vector<node> &X, ull &res) {
	if (P.empty()) {
		if (X.empty()) res++;
		return;
	}
	node u = P[0];

	int vit1 = 0;
	int vit2 = csr_vlist[u];
    int ud = csr_vlist[u+1];
	int ps = P.size();
	int xs = X.size();

	while (vit1 != ps) {
		if (vit2 == ud || P[vit1] < csr_elist[vit2]) {
			int v = P[vit1];
			int gvs = csr_vlist[v+1] - csr_vlist[v];
			vector<node> NP;
			NP.reserve(min(ps, gvs));
			vector<node> NX;
			NX.reserve(min(xs, gvs));
			get_common_neis(v, X, NX);
			get_common_neis(v, P, NP);
			BKP(NP, NX, res);
			P.erase(P.begin()+vit1);
			X.insert(lower_bound(X.begin(), X.end(), v), v);
			xs++;
			ps--;
            vector<node>().swap(NP);
            vector<node>().swap(NX);
		}
		else if (P[vit1] == csr_elist[vit2]) {
            vit1++;
            vit2++;
        }
        else vit2++;
	}
}

void Graph::bfs(node u, vector<size_t> &bfs_rank) {
    bfs_rank.clear();
    bfs_rank.resize(num_nodes, UINT_MAX);
    bfs_rank[u] = 0;

    vector<node> frontiers;
    frontiers.reserve(num_nodes);
    frontiers.push_back(u);

    size_t curr_seen_cnt = 0;
    size_t next_seen_cnt = frontiers.size();
    size_t rank = 1;
    while (next_seen_cnt > curr_seen_cnt) {
        for (size_t i = curr_seen_cnt; i < next_seen_cnt; i++) {
            for (size_t j = csr_vlist[frontiers[i]]; j < csr_vlist[frontiers[i] + 1]; j++) {
                node v = csr_elist[j];
                if (bfs_rank[v] == UINT_MAX) {
                    bfs_rank[v] = rank;
                    frontiers.push_back(v);
                }
            }
        }
        curr_seen_cnt = next_seen_cnt;
        next_seen_cnt = frontiers.size();
        rank++;
    }
}

void Graph::cc(vector<node> &component_id) {
    component_id.clear();
    component_id.resize(num_nodes, INT_MAX);

    queue<node> frontiers;
    for (node u = 0; u < num_nodes; u++) {
        if (component_id[u] != INT_MAX) continue;
        frontiers.push(u);
        component_id[u] = u;
        while (!frontiers.empty()) {
            node v = frontiers.front();
            frontiers.pop();
            for (size_t i = csr_vlist[v]; i < csr_vlist[v+1]; i++) {
                node w = csr_elist[i];
                if (component_id[w] == INT_MAX) {
                    component_id[w] = u;
                    frontiers.push(w);
                }
            }
        }
    }
}

void Graph::core_decomposition(vector<size_t> &core_numbers) {
    core_numbers.clear();
    core_numbers.resize(num_nodes);
    vector<pair<node, size_t>> degs;
    degs.reserve(num_edges);
    for (node u = 0; u < num_nodes; u++) {
        core_numbers[u] = csr_vlist[u+1] - csr_vlist[u];
        degs.push_back({u, core_numbers[u]});
    }

    make_heap(degs.begin(), degs.end(), weight_greater<size_t>);

    while (!degs.empty()) {
        pop_heap(degs.begin(), degs.end(), weight_greater<size_t>);
        node u = degs.back().first;
        size_t _core_number = degs.back().second;
        degs.pop_back();
        if (core_numbers[u] < _core_number) continue;
        for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
            node v = csr_elist[i];
            if (core_numbers[v] <= core_numbers[u]) continue;
            core_numbers[v]--;
            degs.push_back({v, core_numbers[v]});
            push_heap(degs.begin(), degs.end(), weight_greater<size_t>);
        }
    }
}

void Graph::dfs_helper(node u, vector<bool> &visited, vector<int> &new2origin) const {
    visited[u] = true;
    new2origin.emplace_back(u);
    for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
        node v = csr_elist[i];
        if (visited[v] == false) dfs_helper(v, visited, new2origin);
    }
}

void Graph::get_dfs_order(node u, vector<int> &new2origin) const {
    vector<bool> visited(num_nodes, false);
    new2origin.reserve(num_nodes);
    dfs_helper(u, visited, new2origin);
}

void Graph::page_rank(vector<double> &res, double epsilon, size_t max_iter) {
	res = vector<double>(num_nodes, (double) 1);
	vector<double> tmp_res(num_nodes, 0);
	double diff = (double) num_nodes;
	size_t iter = 0;
	while (diff > epsilon && iter < max_iter) {
	// while (iter < max_iter) {
		iter++;
		diff = 0;
		swap(res, tmp_res);
        for (size_t u = 0; u < num_nodes; u++) {
            res[u] = 0;
			if (csr_vlist[u] == csr_vlist[u+1]) continue;
			for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
                node v = csr_elist[i];
                res[u] += tmp_res[v] / (csr_vlist[v+1] - csr_vlist[v]);
			}
            diff += fabs(res[u] - tmp_res[u]);
		}
	}
	printf("-------- PageRank value --------\n");
	for (size_t u = 0; u < 10; u++) {
		printf("Node %lu, pr value: %.3lf.\n", u, res[u]);
	}
}

void Graph::subgraph_matching_helper(const Graph &q, unordered_set<node> &partial_matching, vector<int> &join_order, vector<std::vector<node>> &inter_res, ull &res) {
    vector<node> neis;
    node curr_join_idx = partial_matching.size();
    // printf("Current partial matching size: %d.\n", curr_join_idx);
    q.get_neis(curr_join_idx, neis);
    vector<node> join_idx_sequence;
    join_idx_sequence.reserve(neis.size());
    for (node v : neis) {
        if (join_order[v] < join_order[curr_join_idx]) join_idx_sequence.emplace_back(v);
    }
    if (join_idx_sequence.empty()) return;
    get_neis(inter_res[join_idx_sequence[0]].back(), neis);
    for (auto join_idx : join_idx_sequence) {
        if (join_idx == join_idx_sequence[0]) continue;
        get_common_neis(inter_res[join_idx].back(), neis, inter_res[curr_join_idx]);
        swap(neis, inter_res[curr_join_idx]);
    }
    inter_res[curr_join_idx].clear();
    for (auto v : neis) {
        if (partial_matching.count(v)) continue;
        inter_res[curr_join_idx].push_back(v);
    }
    vector<node>().swap(neis);

    if (curr_join_idx == join_order.size() - 1) {
        res += inter_res[curr_join_idx].size();
        return;
    }
    
    while (!inter_res[curr_join_idx].empty()) {
        node v = inter_res[curr_join_idx].back();
        if (!partial_matching.count(v)) {
            partial_matching.insert(v);
            subgraph_matching_helper(q, partial_matching, join_order, inter_res, res);
            partial_matching.erase(v);
        }
        inter_res[curr_join_idx].pop_back();
    }
    return;
}

void Graph::subgraph_matching_helper(const Graph &q, vector<node> &partial_matching, vector<int> &join_order, ull &res) {
    vector<node> neis;
    node curr_join_idx = partial_matching.size();
    // printf("Current partial matching size: %d.\n", curr_join_idx);
    q.get_neis(curr_join_idx, neis);
    vector<node> join_idx_sequence;
    join_idx_sequence.reserve(neis.size());
    for (node v : neis) {
        if (join_order[v] < join_order[curr_join_idx]) join_idx_sequence.emplace_back(v);
    }
    if (join_idx_sequence.empty()) return;
    node min_degree_join_vertex = partial_matching[join_idx_sequence[0]];
    int min_degree_join = get_deg(min_degree_join_vertex);
    for (auto join_idx : join_idx_sequence) {
        int degree_join = get_deg(partial_matching[join_idx]);
        if (degree_join < min_degree_join) {
            min_degree_join = degree_join;
            min_degree_join_vertex = partial_matching[join_idx];
        }
    }
    get_neis(min_degree_join_vertex, neis);
    vector<node> candidates;
    candidates.reserve(neis.size());

    if (join_idx_sequence.size() == 1) {
        swap(neis, candidates);
    }
    else {
        if (curr_join_idx == join_order.size() - 1) {
            node end_join_idx = (min_degree_join_vertex == partial_matching[join_idx_sequence.back()]) ? join_idx_sequence[join_idx_sequence.size()-2] : join_idx_sequence.back();
            for (auto join_idx : join_idx_sequence) {
                if (partial_matching[join_idx] == min_degree_join_vertex) continue;
                if (join_idx == end_join_idx) {
                    res += cnt_common_neis(partial_matching[join_idx], neis);
                    for (node u : partial_matching) {
                        if (find(neis.begin(), neis.end(), u) != neis.end() && has_directed_edge(partial_matching[join_idx], u))
                            res--;
                    }
                    // get_common_neis(partial_matching[join_idx], tmp, candidates);
                    // res += candidates.get_decoded_size(partial_matching);
                    return;
                }
                else {
                    get_common_neis(partial_matching[join_idx], neis, candidates);
                    swap(neis, candidates);
                }    
            }
        }
        else {
            for (auto join_idx : join_idx_sequence) {
                if (partial_matching[join_idx] == min_degree_join_vertex) continue;
                get_common_neis(partial_matching[join_idx], neis, candidates);
                swap(neis, candidates);
            }
            swap(neis, candidates);
        }
    }

    // for (auto join_idx : join_idx_sequence) {
    //     if (partial_matching[join_idx] == min_degree_join_vertex) continue;
    //     get_common_neis(partial_matching[join_idx], neis, candidates);
    //     swap(neis, candidates);
    // }
    // candidates.clear();
    // for (auto v : neis) {
    //     bool is_duplicate = false;
    //     for (auto u : partial_matching){
    //         if (u == v) {
    //             is_duplicate = true;
    //             break;
    //         }
    //     }
    //     if (!is_duplicate) candidates.push_back(v);
    // }
    // vector<node>().swap(neis);

    // if (curr_join_idx == join_order.size() - 1) {
    //     res += candidates.size();
    //     return;
    // }

    for (auto v : candidates) {
        if (get_deg(v) < q.get_deg(curr_join_idx)) continue;
        partial_matching.push_back(v);
        subgraph_matching_helper(q, partial_matching, join_order, res);
        partial_matching.pop_back();
    }
    return;
}