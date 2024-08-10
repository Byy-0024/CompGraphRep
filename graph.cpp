#include"Graph.hpp"

Graph::Graph(string path) {
	bool has_zero = false;
	num_nodes = 0;
	num_edges = 0;
	num_iso_nodes = 0;
	FILE *fp = fopen(path.c_str(), "r");
	if (fp == NULL) {
        std::cout << "fail to open " << path << std::endl;
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
			// printf("Line: %s, u: %d, v: %d.\n", line, u, v);
			has_zero = true;
		}
		if (u == v) continue;
		if (u > num_nodes) num_nodes = u;
		if (v > num_nodes) num_nodes = v;
        edge_list.push_back({(node) u, (node) v});
    }    
    fclose(fp);
	if (has_zero) num_nodes++;
	adj_list.resize(num_nodes);  
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
	for (int i = 0; i < num_nodes; ++i){
		if (adj_list[i].empty()) num_iso_nodes++;
		else {
			sort(adj_list[i].begin(), adj_list[i].end());
			vector<node> tmp_neis(adj_list[i]);
			adj_list[i].clear();
			node prev = UINT_MAX;
			for (auto j : tmp_neis) {
				if (j == prev) continue;
				prev = j;
				adj_list[i].push_back(j);
			}
			num_edges += adj_list[i].size();
		}
	}
	vector<pair<node, node>>().swap(edge_list);
	printf("Number of nodes: %lu, number of edges: %lu, size of adj-list: %.3f MB!\n", 
			num_nodes, num_edges, (float)num_edges * sizeof(node) / (1<<20));
}

Graph::Graph(vector<vector<node>> &_adj_list) {
	adj_list = _adj_list;
	num_nodes = _adj_list.size();
	num_edges = 0;
	num_iso_nodes = 0;
	for (auto neis : _adj_list) {
		if (neis.empty()) num_iso_nodes++;
		else num_edges += neis.size();
	}
	printf("Number of nodes: %lu, number of edges: %lu, size of adj-list: %.3f MB!\n", 
			num_nodes - num_iso_nodes, num_edges, (float)num_edges * sizeof(node) / (1<<20));	
}

ull Graph::cnt_common_neighbors(node u, node v) const {
	int u_ind = 0, v_ind = 0;
	ull ans = 0;
	while(u_ind < adj_list[u].size() && v_ind < adj_list[v].size()) {
		// __cnt++;
		if (adj_list[u][u_ind] == adj_list[v][v_ind]) {
			ans++;
			u_ind++;
			v_ind++;
		}
		else if (adj_list[u][u_ind] < adj_list[v][v_ind]) u_ind++;
		else v_ind++;
	}
	return ans;
}

ull Graph::conditional_cnt_common_neighbor(node u, node v, node mu) {
	int u_ind = 0, v_ind = 0;
	ull ans = 0;
	while(u_ind < adj_list[u].size() && v_ind < adj_list[v].size()) {
		// __cnt++;
		if (adj_list[u][u_ind] < mu) u_ind++;
		else if (adj_list[v][v_ind] < mu) v_ind++;
		else if (adj_list[u][u_ind] == adj_list[v][v_ind]) {
			ans++;
			u_ind++;
			v_ind++;
		}
		else if (adj_list[u][u_ind] < adj_list[v][v_ind]) u_ind++;
		else v_ind++;
	}
	return ans;
}

void Graph::get_common_neighbors(node u, node v, vector<node>& res) const {
	int u_ind = 0, v_ind = 0;
	while(u_ind < adj_list[u].size() && v_ind < adj_list[v].size()) {
		if (adj_list[u][u_ind] == adj_list[v][v_ind]) {
			res.emplace_back(adj_list[u][u_ind]);
			u_ind++;
			v_ind++;
		}
		else if (adj_list[u][u_ind] < adj_list[v][v_ind]) u_ind++;
		else v_ind++;
	}
}

void Graph::create_dag(){
	num_edges = 0;
	vector<size_t> degrees;
	for (size_t i = 0; i < adj_list.size(); ++i) {
		degrees.push_back(adj_list[i].size());
	}
	for(int i = 0; i < adj_list.size(); ++i) {
		vector<node> tmp_neis(adj_list[i]);
		adj_list[i].clear();
		for (auto j : tmp_neis) {
			if (degrees[i] < degrees[j]) adj_list[i].push_back(j);
		}	
		if (!adj_list[i].empty()) sort(adj_list[i].begin(), adj_list[i].end());
		num_edges += adj_list[i].size();
	}
}

void Graph::get_common_neighbors(node u, const vector<node> &nodes, vector<node> &res) const {
	res.clear();
	int u_ind = 0, v_ind = 0;
	while (u_ind < adj_list[u].size() && v_ind < nodes.size()) {
		if (adj_list[u][u_ind] == nodes[v_ind]) {
			res.push_back(nodes[v_ind]);
			u_ind++;
			v_ind++;
		}
		else if (adj_list[u][u_ind] < nodes[v_ind]) u_ind++;
		else v_ind++;
	}
}


ull Graph::cnt_common_neighbors(node u, const vector<node> &nodes) const {
	int u_ind = 0, v_ind = 0;
	ull ans = 0;
	while (u_ind < adj_list[u].size() && v_ind < nodes.size()) {
		if (adj_list[u][u_ind] == nodes[v_ind]) {
			ans++;
			u_ind++;
			v_ind++;
		}
		else if (adj_list[u][u_ind] < nodes[v_ind]) u_ind++;
		else v_ind++;
	}
	return ans;	
}

void Graph::get_neighbors(node u, vector<node>& neis) const {
	if (!neis.empty()) neis.clear();
	if (adj_list[u].empty()) return;
	neis.insert(neis.end(), adj_list[u].begin(), adj_list[u].end());
}

void Graph::get_unseen_neighbors(node u, vector<bool> &seen, vector<node> &neis) const {
	if (!neis.empty()) neis.clear();
	for (auto u : adj_list[u]) {
		if (!seen[u]) neis.emplace_back(u);
	}
}

void Graph::get_two_hop_neis(node u, vector<node> &res) const {
	res.clear();
	vector<bool> seen(num_nodes, false);
	for(auto v : adj_list[u]) {
		for (auto k : adj_list[v]) {
			if (seen[k]) continue;
			res.emplace_back(k);
		}
	}
	if (!res.empty()) sort(res.begin(), res.end());
}

node Graph::get_one_neighbor(node u, int k) const {
	return adj_list[u][k];
}

size_t Graph::get_number_of_neighbors(node u) const {
	return adj_list[u].size();
}

size_t Graph::get_num_nodes() const{
	return num_nodes;
}

size_t Graph::get_num_edges() const{
	return num_edges;
}

size_t Graph::size_in_bytes() const {
	return (num_nodes + num_edges) * sizeof(node);
}

bool Graph::has_directed_edge(node u, node v) const {
	return binary_search(adj_list[u].begin(), adj_list[u].end(), v);
}

bool Graph::has_undirected_edge(node u, node v) const {
	if (adj_list[u].size() < adj_list[v].size()) return binary_search(adj_list[u].begin(), adj_list[u].end(), v);
	else return binary_search(adj_list[v].begin(), adj_list[v].end(), u);
}

ull Graph::cnt_tri_merge() {
	ull res = 0;
	for (int i = 0; i < num_nodes; ++i) {
		if (adj_list[i].empty()) continue;
		for (auto j : adj_list[i]) {
			res += cnt_common_neighbors(i, j);
		}
	}
	return res;
}

void Graph::print_neis(node u) {
	printf("N(%lu): ", u);
	for (auto v : adj_list[u]) {
		printf("%lu ", v);
	}
	printf(".\n");
}

void Graph::reorder(vector<node> &origin_to_new){
	vector<bool> seen(num_nodes, false);
	vector<vector<node>> tmp_adj_list(adj_list);
	for(int i = 0; i < num_nodes; ++i) {
		node new_id = origin_to_new[i];
		if (new_id == UINT_MAX) continue;
		if (seen[new_id]) printf("Error on node %d! Already has node with label %lu!\n", i, new_id);
		else seen[new_id] = true;
		adj_list[new_id].clear();
		for (auto j : tmp_adj_list[i]) {
			adj_list[new_id].push_back(origin_to_new[j]);
		}	
		if (!adj_list[new_id].empty()) sort(adj_list[new_id].begin(), adj_list[new_id].end());
	}
}

void Graph::save(string path) {
	ofstream out(path);
	for (int i = 0; i < num_nodes; i++) {
		for (auto j : adj_list[i]) {
			if (i < j) out << i  << " " << j  << endl;
		}
	}
	out.close();
}

void Graph::bfs(node u, vector<int> &bfs_rank) {
	node curr = u;
	bfs_rank[curr] = 0;
	vector<node> frontier;
	frontier.reserve(num_nodes);
	frontier.push_back(curr);
    int rank = 1;
	size_t curr_seen_cnt = 0;
	size_t next_seen_cnt = frontier.size();
    while (next_seen_cnt > curr_seen_cnt) {
		for (size_t i = curr_seen_cnt; i < next_seen_cnt; ++i) {
			curr = frontier[i];
			_mm_prefetch((char *) &adj_list[frontier[i+1]], _MM_HINT_T1);
            for (auto j : adj_list[curr]) {
				if (bfs_rank[j] == INT_MAX) {
					frontier.push_back(j);
					bfs_rank[j] = rank;
				}
            }
        } 
        rank++;
		curr_seen_cnt = next_seen_cnt;
		next_seen_cnt = frontier.size();
    }
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
			tmp_res[u] /= adj_list[u].size();
		}
		for (size_t u = 0; u < num_nodes; u++) {
			res[u] = 0;
			if (adj_list[u].empty()) continue;
			for (auto v : adj_list[u]) {
				// res[u] += tmp_res[v] / adj_list[v].size();
				res[u] += tmp_res[v];
			}
			diff += fabs(res[u] - tmp_res[u]);
			diff += fabs(res[u] - tmp_res[u] * adj_list[u].size());
		}
	}
	printf("-------- PageRank value --------\n");
	for (size_t u = 0; u < 5; u++) {
		printf("Node %lu, pr value: %.3lf.\n", u, res[u]);
	}
}

WeightedGraph::WeightedGraph(vector<vector<pair<node, size_t>>> &edges) {
	num_nodes = edges.size();
	num_edges = 0;
	adj_list.resize(num_nodes);
	for (size_t u = 0; u < num_nodes; u++) {
		for (auto e : edges[u]) {
			if (e.first == UINT_MAX) continue;
			adj_list[u].push_back(e);
			adj_list[e.first].push_back({ u, e.second });
		}
		num_edges += edges[u].size();
	}
    vector<pair<node, weight>> tmp_adj_list;
	for (size_t u = 0; u < num_nodes; u++) {
		if (adj_list[u].empty()) continue;
		sort(adj_list[u].begin(), adj_list[u].end(), vid_less);
        tmp_adj_list.clear();
		tmp_adj_list.reserve(adj_list[u].size());
		tmp_adj_list.emplace_back(adj_list[u][0]);
		for (size_t i = 1; i < adj_list[u].size(); i++) {
			if (adj_list[u][i].first != tmp_adj_list.back().first) {
				tmp_adj_list.push_back(adj_list[u][i]);
			}
		}
		swap(adj_list[u], tmp_adj_list);
		vector<pair<node, weight>>().swap(tmp_adj_list);
	}
    vector<pair<node, weight>>().swap(tmp_adj_list);
	printf("Simplified graph has been constructed!");
	printf("Num of nodes: %lu, num of edges: %lu.\n", num_nodes, num_edges);
}

WeightedGraph::~WeightedGraph() {
	vector<vector<pair<node, weight>>>().swap(adj_list);
}

size_t WeightedGraph::get_weight(node u, node v) {
	if (adj_list[u].size() > adj_list[v].size()) swap(u, v);
	for (auto e : adj_list[u]) {
		if (e.first == v) return e.second;
	}
}

void WeightedGraph::get_ordering_greedy(vector<node> &origin_to_new) {
	origin_to_new.clear();
	origin_to_new.resize(num_nodes, UINT_MAX);
	for (size_t u = 0; u < num_nodes; u++) {
		if (!adj_list[u].empty()) {
			sort(adj_list[u].begin(), adj_list[u].end(), weight_greater);
		}
	}
	node curr_new_id = 0;
	node curr = 0;
	node first_unseen = 1;
	while (curr_new_id < num_nodes) {
		origin_to_new[curr] = curr_new_id++;
		if (!adj_list[curr].empty()) {
			for (auto e : adj_list[curr]) {
				if (origin_to_new[e.first] == UINT_MAX) {
					curr = e.first;
					break;
				}
			}
			if (origin_to_new[curr] == UINT_MAX) continue; 
		}
		while (origin_to_new[first_unseen] != UINT_MAX) {
			first_unseen++;
			if (first_unseen >= num_nodes) break;
		}
		curr = first_unseen;
	}
}

void WeightedGraph::maximun_cycle_cover_greedy(vector<pair<node, node>> &res) {
	vector<pair<node, weight>> candidates;
	candidates.reserve(num_nodes);
	vector<size_t> head_pos(num_nodes, 0);
	size_t total_weight = 0, edge_cnt = 0;
	for (size_t u = 0; u < num_nodes; u++) {
		if (adj_list[u].empty()) continue;
		sort(adj_list[u].begin(), adj_list[u].end(), weight_greater);
		candidates.push_back({u, adj_list[u][0].second});
	}
	make_heap(candidates.begin(), candidates.end(), weight_less);
	while (!candidates.empty()) {
		pop_heap(candidates.begin(), candidates.end(), weight_less);
		node u = candidates.back().first;
		node v = adj_list[u][head_pos[u]].first;
		weight w = candidates.back().second;
		candidates.pop_back();
		if (res[u].second != UINT_MAX) continue;
		if (res[u].first != v && res[v].second == UINT_MAX) {
			(res[u].first == UINT_MAX ? res[u].first : res[u].second) = v;
			(res[v].first == UINT_MAX ? res[v].first : res[v].second) = u;
			edge_cnt++;
			total_weight += w;
		}
		if (res[u].second == UINT_MAX) {
			// printf("u: %u, res[v].first: %u, res[v].second:%u, head_pos[u]: %lu.\n", u, res[u].first, res[v].second, head_pos[u]);
			while (res[v].second != UINT_MAX || res[v].first == u) {
				head_pos[u]++;
				if (head_pos[u] == adj_list[u].size()) break;
				v = adj_list[u][head_pos[u]].first;
			}
			if (head_pos[u] == adj_list[u].size()) continue;
			candidates.push_back({u, adj_list[u][head_pos[u]].second});
			push_heap(candidates.begin(), candidates.end(), weight_less);
		}
	}
	vector<pair<node, weight>>().swap(candidates);
	vector<size_t>().swap(head_pos);
	// printf("Approximate maximum cycle cover: %lu edges with weight %lu .\n", edge_cnt, total_weight);
}

void WeightedGraph::maximum_perfect_matching_greedy(vector<pair<node, node>> &res) {
	res.clear();
	res.resize(num_nodes, {UINT_MAX, UINT_MAX});
	vector<size_t> header_pos(adj_list.size(), 0);
	vector<tuple<node, node, weight>> candidates;
	weight total_weight = 0;
	size_t _edge_cnt = 0;
	for (node u = 0; u < adj_list.size(); u++) {
		if (adj_list[u].empty()) continue;
		sort(adj_list[u].begin(), adj_list[u].end(), weight_greater);
		candidates.push_back(tuple<node, node, weight>( u, adj_list[u][0].first, adj_list[u][0].second ));
		header_pos[u]++;
	}
	make_heap(candidates.begin(), candidates.end(), [](const tuple<node, node, weight>& x, const tuple<node, node, weight>& y) {
		return get<2>(x) < get<2>(y);
	});
	while (!candidates.empty()) {
		pop_heap(candidates.begin(), candidates.end(), [](const tuple<node, node, weight>& x, const tuple<node, node, weight>& y) {
			return get<2>(x) < get<2>(y);
		});
		node u = get<0>(candidates.back()), v = get<1>(candidates.back());
		weight w = get<2>(candidates.back());
		candidates.pop_back();
		if (res[u].first != UINT_MAX) continue;
		if (res[v].first != UINT_MAX) {
			if (header_pos[u] == adj_list[u].size()) continue;
			candidates.push_back(tuple<node, node, weight>( u, adj_list[u][header_pos[u]].first, adj_list[u][header_pos[u]].second ));
			push_heap(candidates.begin(), candidates.end(), [](const tuple<node, node, weight>& x, const tuple<node, node, weight>& y) {
				return get<2>(x) < get<2>(y);
			});
			header_pos[u]++;
			continue;
		}
		else {
			res[u].first = v;
			res[v].first = u;
			total_weight += w;
			_edge_cnt++;
		}
	}
	// printf("Approximate maximum weight matching on general graph: %lu edges with weight %lu .\n", _edge_cnt, total_weight);
}

void WeightedGraph::expand_cycle_cover_to_ordering(vector<pair<node, node>> &CC, vector<node> &origin_to_new) {
	vector<node> path_headers;
	vector<node> cycle_headers;
	vector<bool> seen(num_nodes, false);
	for (size_t i = 0; i < CC.size(); i++) {
		if (seen[i] == true) continue;
		seen[i] = true;
		node prev = i, next = CC[i].second;
		while (true) {
			if (next == UINT_MAX) {
				path_headers.push_back(i);
				break;
			}
			if (seen[next]) {
				for (size_t j = 0; j < path_headers.size(); j++) {
					if (path_headers[j] == next) {
						path_headers[j] = i;
						break;
					}
				}
				break;
			}
			else {
				seen[next] = true;
				if (CC[next].second == prev) swap(CC[next].first, CC[next].second);
				prev = next; 
				next = CC[next].second;
				if (next == i) {
					cycle_headers.emplace_back(i);
					break;
				}
			}
		}
	}
	// printf("Num of cycles: %lu, num of paths: %lu.\n", cycle_headers.size(), path_headers.size());
	size_t weight_of_remove_edges = 0;
	for (size_t i = 0; i < cycle_headers.size(); i++) {
		node u = cycle_headers[i];
		node v = CC[u].second;
		node remove_u = u, remove_v = v;
		size_t minimum_weight = get_weight(u, v);
		while (v != cycle_headers[i]) {
			u = v;
			v = CC[u].second;
			if (get_weight(u, v) < minimum_weight) {
				remove_u = u, remove_v = v;
				minimum_weight = get_weight(u, v);
			}
		}
		weight_of_remove_edges += minimum_weight;
		CC[remove_u].second = UINT_MAX;
		path_headers.push_back(remove_v);
	}
	// printf("Total weight of removed edges: %lu.\n", weight_of_remove_edges);
	seen.clear();
	seen.resize(num_nodes, true);
	for (auto u : path_headers) {
		seen[u] = false;
	}

	origin_to_new.clear();
	origin_to_new.resize(num_nodes, UINT_MAX);
	node curr_new_id = 0;
	node curr = path_headers[0];
	size_t header_ptr = 1;
	while (curr_new_id < num_nodes) {
		seen[curr] = true;
		origin_to_new[curr] = curr_new_id++;
		while (CC[curr].second != UINT_MAX) {
			curr = CC[curr].second;
			origin_to_new[curr] = curr_new_id++;
		}
		// curr = arg_max_common_neighbors(curr, seen);
		for (auto e : adj_list[curr]) {
			if (!seen[e.first]) {
				curr = e.first;
				break;
			}
		}
		if (CC[curr].second == UINT_MAX) {
			while (seen[path_headers[header_ptr]]) {
				header_ptr++;
				if (header_ptr >= path_headers.size()) break;
			}
			curr = path_headers[header_ptr];
		}
	}
}

void WeightedGraph::expand_perfect_matching_to_ordering(vector<pair<node, node>> &GMWM, vector<node> &origin_to_new) {
	node head = UINT_MAX, curr_new_id = 0;
	size_t max_edge_weight = 0;
	origin_to_new.clear();
	origin_to_new.resize(num_nodes, UINT_MAX);
	for (size_t i = 0; i <  adj_list.size(); i++) {
		if (adj_list[i].empty()) continue;
		if (adj_list[i][0].second > max_edge_weight) {
			head = i;
			max_edge_weight = adj_list[head][0].second;
		}
	}
	node curr = head;
	node prev = UINT_MAX;
	node first_unseen = 0;
	while (curr_new_id < adj_list.size()) {
		origin_to_new[curr] = curr_new_id++;
		node next = GMWM[curr].first;
		if (next == UINT_MAX) {
			for (auto e : adj_list[curr]) {
				if (origin_to_new[e.first] == UINT_MAX) {
					next = e.first;
					break;
				}
			}
			if (next == UINT_MAX) {
				for (first_unseen; first_unseen < adj_list.size(); first_unseen++) {
					if (origin_to_new[first_unseen] == UINT_MAX) {
						next = first_unseen;
						break;
					}
				}
				if (next == UINT_MAX) break;
			}
		}
		else if (next == prev) {
			for (auto e : adj_list[curr]) {
				if (origin_to_new[e.first] == UINT_MAX) {
					next = e.first;
					break;
				}
			}
			if (next == prev) {
				for (first_unseen; first_unseen < adj_list.size(); first_unseen++) {
					if (origin_to_new[first_unseen] == UINT_MAX) {
						next = first_unseen;
						break;
					}
				}
				if (next == prev) break;
			}			
		}
		prev = curr;
		curr = next;
	}
	for (size_t i = 0; i < adj_list.size(); i++) {
		if (origin_to_new[i] == UINT_MAX) {
			origin_to_new[i] = curr_new_id++;
		}
	}
}