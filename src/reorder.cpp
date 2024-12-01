#include "reorder.hpp"

using namespace std;

void save_ordering_as_binfile(const vector<node> &ordering, const string &binfilename) {
    ofstream fout(binfilename, ios::binary);
    fout.write((char*)ordering.data(), sizeof(node) * ordering.size());
    fout.close();
}

WeightedGraph::WeightedGraph(vector<vector<pair<node, weight>>> &edges) {
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
		sort(adj_list[u].begin(), adj_list[u].end(), vid_less<weight>);
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

weight WeightedGraph::get_weight(node u, node v) {
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
			sort(adj_list[u].begin(), adj_list[u].end(), weight_greater<weight>);
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
		sort(adj_list[u].begin(), adj_list[u].end(), weight_greater<weight>);
		candidates.push_back({u, adj_list[u][0].second});
	}
	make_heap(candidates.begin(), candidates.end(), weight_less<weight>);
	while (!candidates.empty()) {
		pop_heap(candidates.begin(), candidates.end(), weight_less<weight>);
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
			push_heap(candidates.begin(), candidates.end(), weight_less<weight>);
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
		sort(adj_list[u].begin(), adj_list[u].end(), weight_greater<weight>);
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

unordered_set<string> ordering_methods = {"Greedy", "MWPM", "Serdyukov"};

int main(int argc, char **argv) {
    string graphdict = argv[1];
    string orderingmethod = argv[2];
    if (!ordering_methods.count(orderingmethod)) {
        printf("Invalid ordering method! Please select from Greedy, MWPM, and Serdyukov.\n");
        return 0;
    }
    // string graphdict = get_graphdict(graphname);
    // string outputdict = get_outputdict(graphname, orderingmethod);
    time_counter _t;
    _t.start();
    Graph G(graphdict+"/origin/csr_vlist.bin", graphdict + "/origin/csr_elist.bin");
    _t.stop();
    printf("Has read graph from %s.\n", graphdict.c_str());
    _t.print("graph construction");  

	HVI g_hvi_old(G);
    ull _size = g_hvi_old.size_in_bytes();
    printf("Size of HVI with origin ordering: %.3f MB!\n", (float)_size / (1 << 20));
    g_hvi_old.save_as_binfile(graphdict + "/origin/hvi_offsets.bin", graphdict + "/origin/hvi_list.bin");

    size_t num_nodes = G.get_num_nodes();
    size_t deg_upper_bound = (size_t)(G.get_num_edges() / num_nodes) * 5;
    vector<vector<pair<node, weight>>> edges;
    edges.resize(num_nodes); 

    size_t _per = num_nodes / 100, _per_cnt = 0;
    srand(chrono::system_clock::now().time_since_epoch().count());
    vector<node> rands(2000);
    for (size_t i = 0; i < 2000; i++) {
        rands[i] = rand();
    }
    _t.clean();
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
            size_t deg = G.get_deg(i);
            seen[_tid][i] = true;
            candidates[_tid].clear();
            size_t num_samples = min((size_t) 1000, 5 * deg);
            size_t _res_cnt = 0, _can_cnt = 0;
            for (size_t j = 0; j < num_samples; j++) {
                node v = G.get_nei(i, rands[(i+j*2) % 2000] % deg);
                node w = G.get_nei(v, rands[(i+2*j+1) % 2000] % G.get_deg(v));
                if (!seen[_tid][w]) {
                    candidates[_tid][_can_cnt++] = w;
                    seen[_tid][w] = true;
                }
            }
            if (_can_cnt == 0) continue;
            for (size_t k = 0; k < _can_cnt; k++) {
                res[_tid][_res_cnt++] = {candidates[_tid][k], G.cnt_common_neis(i, candidates[_tid][k])};
            }
            sort(res[_tid].begin(), res[_tid].begin() + min(deg_upper_bound, _res_cnt), weight_greater<weight>);
            edges[i].insert(edges[i].begin(), res[_tid].begin(), res[_tid].begin() + min(deg_upper_bound, _res_cnt));
        }
    }
    WeightedGraph WG(edges);
    _t.stop();
    _t.print("simplified graph construction");
    vector<vector<node>>().swap(candidates);
    vector<vector<bool>>().swap(seen);
    vector<vector<pair<node, weight>>>().swap(res);
    vector<vector<pair<node, weight>>>().swap(edges);

    vector<node> origin_to_new;
    _t.clean();
    _t.start();
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
    _t.stop();
    _t.print("reordering");
	save_ordering_as_binfile(origin_to_new, graphdict + "/" + orderingmethod + "/origin2new.bin");
    G.reorder(origin_to_new);
    G.save_as_csr_binfile(graphdict + "/" + orderingmethod + "/csr_vlist.bin", graphdict + "/" + orderingmethod + "/csr_elist.bin");
    _size = G.size_in_bytes();
    printf("Size of CSR with new ordering: %.3f MB!\n", (float)_size / (1 << 20));
    HVI g_hvi(G);
    _size = g_hvi.size_in_bytes();
    printf("Size of HVI with new ordering: %.3f MB!\n", (float)_size / (1 << 20));
    g_hvi.save_as_binfile(graphdict + "/" + orderingmethod + "/hvi_offsets.bin", graphdict + "/" + orderingmethod + "/hvi_list.bin");
	return 0;
}