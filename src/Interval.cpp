#include "Interval.hpp"

using namespace std;

bool hybrid_vid_less(const HybridVertexInterval &x, const HybridVertexInterval &y) {
    return (x & GETNODE_MASK) < (y & GETNODE_MASK);
}

bool hybrid_vid_greater(const HybridVertexInterval &x, const HybridVertexInterval &y) {
    return (x & GETNODE_MASK) > (y & GETNODE_MASK);
}

UI::UI(const Graph &g) {
    num_nodes = g.get_num_nodes();
    num_edges = g.get_num_edges();
    inds = (size_t *)malloc((num_nodes+1) * sizeof(size_t));
    vector<Interval> tmp_interval_list;
    tmp_interval_list.reserve(num_nodes);
    vector<node> neis;
    for (node u = 0; u < num_nodes; ++u) {
        g.get_neis(u, neis);
        inds[u] = tmp_interval_list.size();
        node left = neis[0];
        node right = left;
        for (auto v : neis) {
            if (v == right) {
                right++;
            }
            else {
                tmp_interval_list.push_back((Interval) left << 32 + right);
                left = v;
                right = v+1;
            }
        }
        tmp_interval_list.push_back((Interval) left << 32 + right);
    }
    inds[num_nodes] = tmp_interval_list.size();
    intervals = (Interval*)malloc(inds[num_nodes] * sizeof(Interval));
    memcpy(intervals, tmp_interval_list.data(), inds[num_nodes] * sizeof(Interval));
}

size_t UI::size_in_bytes() const {
    return inds[num_nodes] * sizeof(Interval);
}

node hvi_iterator::get_val() const{
        return curr_val;
    }

hvi_iterator hvi_iterator::operator++() {
    if (curr_val < right_val) {
        curr_val++;
    }
    else {
        curr_pos ++;
        curr_val = curr_pos[0] & GETNODE_MASK;
        if (curr_pos[0] & HVI_LEFT_BOUNDARY_MASK) {
            curr_pos++;
            right_val = curr_pos[0] & GETNODE_MASK;
        }
        else {
            right_val = curr_val;
        }
    }
    return *this;
} 

hvi_iterator::hvi_iterator(const HVI &g_hvi, node u) {
    curr_pos = g_hvi.get_neis(u);
    end_pos = g_hvi.get_neis(u+1);
    if (end_pos[0] & HVI_LEFT_BOUNDARY_MASK) end_pos++;
    if (curr_pos == end_pos) return;
    curr_val = curr_pos[0] & GETNODE_MASK;
    if (curr_pos[0] & HVI_LEFT_BOUNDARY_MASK) {
        curr_pos++;
        right_val = curr_pos[0] &GETNODE_MASK;
    }
    else right_val = curr_val;
}

// node hvi_iterator::get_val() const{
//         return curr_val;
//     }

// hvi_iterator hvi_iterator::operator++() {
//     if (step) {
//         curr_val ++;
//         step --;
//     }
//     else {
//         curr_pos ++;
//         curr_val = curr_pos[0] & GETNODE_MASK;
//         step = (curr_pos[0] & HVI_LEFT_BOUNDARY_MASK) ? (curr_pos[1] & GETNODE_MASK) - curr_val - 1: 0;
//     }
//     return *this;
// } 

// hvi_iterator::hvi_iterator(const HVI &g_hvi, node u) {
//     curr_pos = g_hvi.get_neis(u);
//     end_pos = g_hvi.get_neis(u+1);
//     if (curr_pos == end_pos) return;
//     curr_val = curr_pos[0] & GETNODE_MASK;
//     step = (curr_pos[0] & HVI_LEFT_BOUNDARY_MASK) ? (curr_pos[1] & GETNODE_MASK) - curr_val : 0;
// }

hvi_vector::hvi_vector(const std::vector<HybridVertexInterval> &hvis) {
    hvi_list = vector<HybridVertexInterval>(std::move(hvis));
    curr_back = hvi_list.empty() ? 0 : (hvi_list.back() & GETNODE_MASK);
    // printf("curr_back: %d.\n", curr_back);
}

bool hvi_vector::empty() const {
    return hvi_list.empty();
}

bool hvi_vector::has_element(node u) const {
    if (hvi_list.empty()) return false;
    size_t lower_bound_pos = lower_bound(hvi_list.begin(), hvi_list.end(), u, hybrid_vid_less) - hvi_list.begin();
    if (lower_bound_pos == hvi_list.size()) return false;
    if (hvi_list[lower_bound_pos] & HVI_RIGHT_BOUNDARY_MASK) {
        return true;
    }
    else {
        return (hvi_list[lower_bound_pos] & GETNODE_MASK) == u;
    }
}

node hvi_vector::back() const {
    return curr_back;
}

size_t hvi_vector::get_decoded_size() const {
    if (hvi_list.empty()) return 0;
    size_t res = 0;
    for (size_t pos = 0; pos < hvi_list.size();) {
        if (hvi_list[pos] & HVI_LEFT_BOUNDARY_MASK) {
            res += (hvi_list[pos+1] & GETNODE_MASK) - (hvi_list[pos] & GETNODE_MASK) + 1;
            pos += 2;
        }
        else {
            res++;
            pos++;
        }
    }
    return res;
}

size_t hvi_vector::get_decoded_size(vector<node> &nodes) const {
    // if (hvi_list.empty()) return 0;
    // size_t res = 0, pos_nodes = 0;
    // for (size_t pos = 0; pos < hvi_list.size();) {
    //     if (hvi_list[pos] & HVI_LEFT_BOUNDARY_MASK) {
    //         node left = (hvi_list[pos] & GETNODE_MASK), right = (hvi_list[pos+1] & GETNODE_MASK);
    //         res += (right - left + 1);
    //         pos += 2;
    //         while (pos_nodes < nodes.size() && nodes[pos_nodes] < left) pos_nodes++;
    //         while (pos_nodes < nodes.size() && nodes[pos_nodes] <= right) {
    //             res--;
    //             pos_nodes++;
    //         }
    //     }
    //     else {
    //         while (pos_nodes < nodes.size() && nodes[pos_nodes] < hvi_list[pos]) pos_nodes++;
    //         if (hvi_list[pos] != nodes[pos_nodes]) res++;
    //         pos++;
    //     }
    // }
    // return res;
    if (hvi_list.empty()) return 0;
    size_t res = 0;
    for (size_t pos = 0; pos < hvi_list.size();) {
        if (hvi_list[pos] & HVI_LEFT_BOUNDARY_MASK) {
            res += (hvi_list[pos+1] & GETNODE_MASK) - (hvi_list[pos] & GETNODE_MASK) + 1;
            pos += 2;
        }
        else {
            res++;
            pos++;
        }
    }
    for (auto u : nodes) {
        if (has_element(u)) res--;
    }
    return res;
}

size_t hvi_vector::get_decoded_size(unordered_set<node> &nodes) const {
    if (hvi_list.empty()) return 0;
    size_t res = 0;
    for (size_t pos = 0; pos < hvi_list.size();) {
        if (hvi_list[pos] & HVI_LEFT_BOUNDARY_MASK) {
            res += (hvi_list[pos+1] & GETNODE_MASK) - (hvi_list[pos] & GETNODE_MASK) + 1;
            pos += 2;
        }
        else {
            res++;
            pos++;
        }
    }
    for (auto u : nodes) {
        if (has_element(u)) res--;
    }
    return res;
}

size_t hvi_vector::size() const {
    return hvi_list.size();
}

void hvi_vector::clear() {
    hvi_list.clear();
    curr_back = 0;
}

// void hvi_vector::erase(node u) {
//     size_t _pos = lower_bound(hvi_list.begin(), hvi_list.end(), u, hybrid_vid_less) - hvi_list.begin();
//     if (_pos == hvi_list.size()) return;
//     if (hvi_list[_pos] & HVI_LEFT_BOUNDARY_MASK) {
//         if (hvi_list[_pos] & GETNODE_MASK == u) {
//             hvi_list[_pos]++;
//         }
//         return;
//     }
//     else if (hvi_list[_pos] & HVI_RIGHT_BOUNDARY_MASK) {
//         node val = hvi_list[_pos] & GETNODE_MASK;
//         if (val == u) {
//             hvi_list[_pos]--;
//             return;
//         }
//         else {
            
//         }
//     }
//     else {
//         hvi_list.erase(hvi_list.begin() + _pos);
//     }
// }

void hvi_vector::pop() {
    HybridVertexInterval hvi_back = hvi_list.back();
    // printf("curr_back: %d, hvi_back: %d.\n", curr_back, (node)(hvi_back & GETNODE_MASK));
    // printf("hvi_vector.size(): %lu.\n", hvi_list.size());
    if (hvi_back & HVI_RIGHT_BOUNDARY_MASK) {
        curr_back--;
        hvi_list.pop_back();
    }
    else if (hvi_back & HVI_LEFT_BOUNDARY_MASK) {
        if (curr_back <= (node)(hvi_back & GETNODE_MASK)) {
            hvi_list.pop_back();
            curr_back = hvi_list.back() & GETNODE_MASK;
        }
        else curr_back--;
    }
    else {
        hvi_list.pop_back();
        curr_back = hvi_list.back() & GETNODE_MASK;
    }
}

void hvi_vector::print() {
    for (auto x : hvi_list) {
        printf("%u ", x);
    }
    printf("\n");
}

HVI::HVI(const Graph &g) {
    num_nodes = g.get_num_nodes();
    num_hvis = 0;
    hvi_offsets = (size_t*)malloc(sizeof(size_t)*(num_nodes + 1));
    vector<HybridVertexInterval> tmp_hvi_list;
    tmp_hvi_list.reserve(num_nodes);
    size_t hvi_offsets_tail_pos = 0;
    vector<node> neis;
    for (size_t i = 0; i < num_nodes; ++i) {
        hvi_offsets[hvi_offsets_tail_pos++]=tmp_hvi_list.size();
        g.get_neis(i, neis);
        if (neis.empty()) continue;
        node curr_head = neis[0];
        node expect_node = curr_head;
        int curr_length = 0;
        for (auto v : neis) {
            if (v == expect_node) {
                curr_length++;
                expect_node++;
            }
            else {
                if (curr_length == 1) tmp_hvi_list.push_back(curr_head);
                else if (curr_length == 2) {
                    tmp_hvi_list.push_back(curr_head);
                    tmp_hvi_list.push_back(expect_node-1);
                }
                else {
                    tmp_hvi_list.push_back(curr_head | HVI_LEFT_BOUNDARY_MASK);
                    tmp_hvi_list.push_back((expect_node - 1) | HVI_RIGHT_BOUNDARY_MASK);
                }
                curr_head = v;
                expect_node = v+1;
                curr_length = 1;
            }
        }
        if (curr_length == 1) tmp_hvi_list.push_back(curr_head);
        else {
            tmp_hvi_list.push_back(curr_head | HVI_LEFT_BOUNDARY_MASK);
            tmp_hvi_list.push_back((expect_node - 1) | HVI_RIGHT_BOUNDARY_MASK);
        }
    }
    hvi_offsets[hvi_offsets_tail_pos++]=tmp_hvi_list.size();
    num_hvis = tmp_hvi_list.size();
    hvi_list = (HybridVertexInterval*)malloc(sizeof(HybridVertexInterval) * num_hvis);
    memcpy(hvi_list, tmp_hvi_list.data(), sizeof(HybridVertexInterval) * hvi_offsets[num_nodes]);
}

HVI::HVI(const string hvi_offsets_binfile, const string hvi_list_binfile) {
    ifstream input_hvi_offsets(hvi_offsets_binfile, ios::binary);
    if (!input_hvi_offsets.is_open()) {
        cerr << "Unable to open hvi offsets binfile" << endl;
        return;
    }
    input_hvi_offsets.seekg(0, ios::end);
    streamsize size = input_hvi_offsets.tellg();
    input_hvi_offsets.seekg(0, ios::beg);
    hvi_offsets = (size_t *)malloc(size);
    input_hvi_offsets.read(reinterpret_cast<char*>(hvi_offsets), size);
    input_hvi_offsets.close();
    num_nodes = size / sizeof(size_t) - 1;

    ifstream input_hvi_list(hvi_list_binfile, ios::binary);
    if (!input_hvi_list.is_open()) {
        cerr << "Unable to open hvi list binfile" << endl;
        return;
    }
    input_hvi_list.seekg(0, ios::end);
    size = input_hvi_list.tellg();
    input_hvi_list.seekg(0, ios::beg);
    hvi_list = (HybridVertexInterval*)malloc(size);
    input_hvi_list.read(reinterpret_cast<char*>(hvi_list), size);
    input_hvi_list.close();
    num_hvis = size / sizeof(HybridVertexInterval);
    printf("Has read graph! #Vertices: %lu, #HVIs: %lu.\n", num_nodes, num_hvis);
}


// return if node u is contained in the node v's neighbors.
bool HVI::bsearch(node u, node v, size_t pos) const {
    if (hvi_offsets[v] == hvi_offsets[v+1] || (hvi_list[hvi_offsets[v]] & GETNODE_MASK) > u) {
		pos = 0;
		return false;
	}
    if (hvi_list[hvi_offsets[v+1]-1] & GETNODE_MASK < u) {
		pos = hvi_offsets[v+1] - 1;
		return false;
	}
	int left = hvi_offsets[v], right = hvi_offsets[v+1] - 1;
	while (right - left > 1) {
		int tmp = (left + right) >> 1;
		if (hvi_list[tmp] & GETNODE_MASK == u) {
			pos = tmp;
			return true;
		}
		else if (hvi_list[tmp] & GETNODE_MASK < u) left = tmp;
		else right = tmp;
	}
	pos = left;
	return hvi_list[pos] & HVI_LEFT_BOUNDARY_MASK;
}

bool HVI::has_directed_edge(const node u, const node v) const {
    if (hvi_offsets[u+1] == hvi_offsets[u]) return false;
    size_t lower_bound_pos = lower_bound(hvi_list+hvi_offsets[u], hvi_list+hvi_offsets[u+1], v, hybrid_vid_less) - hvi_list;
    if (lower_bound_pos == hvi_offsets[u+1]) return false;
    if (hvi_list[lower_bound_pos] & HVI_RIGHT_BOUNDARY_MASK) {
        return true;
    }
    else {
        return (hvi_list[lower_bound_pos] & GETNODE_MASK) == v;
    }
}

bool HVI::has_edge(node u, node v) const {
    if (hvi_offsets[u] == hvi_offsets[u+1] || hvi_offsets[v] == hvi_offsets[v+1]) return false;
    if (hvi_offsets[u+1] - hvi_offsets[u] < hvi_offsets[v+1] - hvi_offsets[v]) {
        return has_directed_edge(u, v);
    }
    else {
        return has_directed_edge(v, u);
    }
}

size_t HVI::get_num_nodes() const {
    return num_nodes;
}

size_t HVI::get_num_hvi() const {
    return num_hvis;
}

size_t HVI::get_degree(node u) const {
    if (u >= num_nodes) return 0;
    size_t res = 0;
    for (size_t pos = hvi_offsets[u]; pos < hvi_offsets[u+1];) {
        if (hvi_list[pos] & HVI_LEFT_BOUNDARY_MASK) {
            res += (hvi_list[pos+1] & GETNODE_MASK) - (hvi_list[pos] & GETNODE_MASK) + 1;
            pos += 2;
        }
        else {
            res++;
            pos++;
        }
    }
    return res;
}

size_t HVI::size_in_bytes() const {
    return (num_hvis + num_nodes) * sizeof(HybridVertexInterval);
}

size_t* HVI::get_hvi_offsets() const {
    return hvi_offsets;
}

HybridVertexInterval* HVI::get_hvi_list() const {
    return hvi_list;
}

HybridVertexInterval* HVI::get_neis(node u) const {
    return &hvi_list[hvi_offsets[u]];
}

void HVI::get_neis(node u, vector<node> &neis) const {
    neis.clear();
    hvi_iterator iter(*this, u);
    for (iter; iter.end_pos != iter.curr_pos; ++iter) {
        neis.emplace_back(iter.curr_val);
    }
    // memcpy(neis.data(), get_neis(u), sizeof(HybridVertexInterval) * (hvi_offsets[u+1] - hvi_offsets[u]));
}

void HVI::get_neis(node u, vector<HybridVertexInterval> &neis) const {
    neis = vector<HybridVertexInterval>(hvi_list + hvi_offsets[u], hvi_list + hvi_offsets[u+1]);
}

void HVI::print_neighbor(node u) const {
    for (size_t pos = hvi_offsets[u]; pos < hvi_offsets[u+1]; pos++) {
        HybridVertexInterval hvi = hvi_list[pos];
        if (hvi & HVI_LEFT_BOUNDARY_MASK) {
            printf(" [%u,", hvi & GETNODE_MASK);
        }
        else if (hvi & HVI_RIGHT_BOUNDARY_MASK) {
            printf(" %u],", hvi & GETNODE_MASK);
        }
        else printf(" %u ", hvi);
    }
    printf("\n");
}

void HVI::save_as_binfile(const string hvi_offset_binfile, const string hvi_list_binfile) const {
    ofstream output_hvi_offset_binfile(hvi_offset_binfile, ios::binary);
    output_hvi_offset_binfile.write(reinterpret_cast<const char*>(hvi_offsets), sizeof(size_t) * (num_nodes + 1));
    printf("Size of hvi_offset: %lu.\n", (num_nodes + 1));
    output_hvi_offset_binfile.close();

    ofstream output_hvi_list_binfile(hvi_list_binfile, ios::binary);
    output_hvi_list_binfile.write(reinterpret_cast<const char*>(hvi_list), sizeof(HybridVertexInterval) * num_hvis);
    printf("Size of hvi_list: %lu.\n", num_hvis);
    output_hvi_list_binfile.close();
}

ull HVI::mc(node u) {
    ull res = 0;
	vector<node> P;
	get_neis(u, P);
	vector<node> X;
	X.reserve(P.size());
	BKP(P, X, res);
	return res;
}

ull HVI::subgraph_matching(const Graph &q) {
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

    vector<size_t> degs(num_nodes, 0);
    for (size_t i = 0; i < num_nodes; i++) {
        degs[i] = get_degree(i);
    }

    // vector<hvi_vector> inter_res(n);
    // unordered_set<node> partial_matching;
    vector<node> partial_matching;
    partial_matching.reserve(n);
    int _per_cnt = (num_nodes - 1) / 100 + 1, _cnt = 0;
    for (node u = 0; u < num_nodes; u++) {
        if (u % _per_cnt == 0){
            printf("\rMatching Process: [%d/100].", ++_cnt);
            fflush(stdout);
        }
        if (degs[u] < q_max_deg) continue;
        // for (hvi_iterator it(*this, u); it.curr_pos != it.end_pos; ++it) {
        //     node v = it.get_val();
        //     partial_matching = {u, v};
        //     subgraph_matching_helper(q, partial_matching, join_order, degs, res);
        //     // inter_res[0].pop();
        //     partial_matching.clear();
        // }
        
        partial_matching.push_back(u);
        subgraph_matching_helper(q, partial_matching, join_order, degs, res);
        partial_matching.clear();
    }
    printf("\rMatching Process: [100/100].\n");
    return res;
}

void HVI::BKP(vector<node> &P, vector<node> &X, ull &res) {
    if (P.empty()) {
		if (X.empty()) res++;
		return;
	}
    // printf("BKP start!\n");
	node u = P[0];

	size_t vit1 = 0;
	size_t ps = P.size();
	size_t xs = X.size();
    hvi_iterator iter(*this, u);

	while (vit1 != ps) {
		if (iter.curr_pos == iter.end_pos || P[vit1] < iter.curr_val) {
			node v = P[vit1];
			vector<node> NP;
			NP.reserve(ps);
			vector<node> NX;
			NX.reserve(xs);
			get_common_neis(v, X, NX);
			get_common_neis(v, P, NP);
			BKP(NP, NX, res);
			P.erase(P.begin()+vit1);
			X.insert(lower_bound(X.begin(), X.end(), v), v);
			xs++;
			ps--;
		}
		else if (P[vit1] == iter.curr_val) {
            vit1++;
            ++iter;
        }
        else ++iter;
	}
}

void HVI::bfs(node u, vector<size_t> &bfs_rank) {
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
            node curr = frontiers[i];
            for (hvi_iterator iter(*this, curr); iter.curr_pos != iter.end_pos; ++iter) {
                node v = iter.curr_val;
                if (bfs_rank[v] == UINT_MAX) {
                    frontiers.push_back(v);
                    bfs_rank[v] = rank;
                }
            }
            // for (size_t j = hvi_offsets[curr]; j < hvi_offsets[curr + 1]; j++) {
            //     node v = hvi_list[j] & GETNODE_MASK;
            //     node step = (hvi_list[j] & HVI_LEFT_BOUNDARY_MASK) ? (hvi_list[j + 1] & GETNODE_MASK) - v : 1;
            //     for (node offset = 0; offset < step; offset++) {
            //         node k = v + offset;
            //         if (bfs_rank[k] == UINT_MAX) {
            //             frontiers.push_back(k);
            //             bfs_rank[k] = rank;
            //         }
            //     }
            // }
        } 
        curr_seen_cnt = next_seen_cnt;
        next_seen_cnt = frontiers.size();
        rank++;
    }
}

void HVI::cc(vector<int> &in_components) {
    in_components.clear();
	in_components.resize(num_nodes, INT_MAX);
	queue<node> frontier;
	for (node u = 0; u < num_nodes; u++) {
		if (in_components[u] != INT_MAX) continue;
		frontier.push(u);
		in_components[u] = u;
		while (!frontier.empty()) {
            node v = frontier.front();
            frontier.pop();
            for (hvi_iterator iter(*this, v); iter.curr_pos != iter.end_pos; ++iter) {
                node j = iter.curr_val;
                if (in_components[j] == INT_MAX) {
                    frontier.push(j);
                    in_components[j] = u;
                }
            }
		}
	}
}

void HVI::core_decomposition(vector<size_t> &core_number) {
    vector<pair<node, size_t>> degrees;
	degrees.reserve(num_nodes);
	core_number.resize(num_nodes);
    size_t num_edges = 0;
	for (node u = 0; u < num_nodes; u++) {
        core_number[u] = get_degree(u);
		degrees.push_back({u, core_number[u]});
        num_edges += core_number[u];
	}
    degrees.reserve(num_edges);
	make_heap(degrees.begin(), degrees.end(), weight_greater<size_t>);
	while(!degrees.empty()) {
		pop_heap(degrees.begin(), degrees.end(), weight_greater<size_t>);
		node u = degrees.back().first;
		size_t w = degrees.back().second;
		degrees.pop_back();
		if (core_number[u] < w) continue;
		for (hvi_iterator iter(*this, u); iter.curr_pos != iter.end_pos; ++iter) {
            node v = iter.curr_val;
			if (core_number[v] > core_number[u]) {
				core_number[v]--;
				degrees.push_back({v, core_number[v]});
				push_heap(degrees.begin(), degrees.end(), weight_greater<size_t>);
			}
		}
	}
}

void HVI::get_common_neis(node u, node v, vector<HybridVertexInterval> &res) const {
    res.clear();
    size_t u_ptr = hvi_offsets[u], v_ptr = hvi_offsets[v];
    while (u_ptr < hvi_offsets[u+1] && v_ptr < hvi_offsets[v+1]) {
        HybridVertexInterval u_hvi = hvi_list[u_ptr], v_hvi = hvi_list[v_ptr];
        if (u_hvi & HVI_LEFT_BOUNDARY_MASK) {
            u_hvi ^= HVI_LEFT_BOUNDARY_MASK;
            uint32_t u_right = hvi_list[u_ptr+1] & GETNODE_MASK;
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvi_list[v_hvi+1] & HVI_RIGHT_BOUNDARY_MASK;
                if (u_right < v_hvi) {
                    u_ptr += 2;
                    continue;
                }
                else if (u_hvi > v_right) {
                    v_ptr += 2;
                    continue;
                }
                else {
                    res.push_back(max(u_hvi, v_hvi) | HVI_LEFT_BOUNDARY_MASK);
                    res.push_back(min(u_right, v_right) | HVI_RIGHT_BOUNDARY_MASK);
                    if (u_right < v_right) u_ptr += 2;
                    else if (u_right > v_right) v_ptr += 2;
                    else {
                        u_ptr += 2;
                        v_ptr += 2;
                    }
                }
            }
            else {
                if (v_hvi < u_hvi) v_ptr++;
                else if (v_hvi >= u_right) u_ptr++;
                else {
                    res.push_back(v_hvi);
                    v_ptr++;
                }
            }
        }
        else {
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvi_list[v_hvi+1] & GETNODE_MASK;
                if (u_hvi < v_hvi) u_ptr++;
                else if (u_hvi > v_right) v_ptr+=2;
                else {
                    res.push_back(u_hvi);
                    u_ptr++;
                }
            }
            else {
                if (u_hvi < v_hvi) u_ptr++;
                else if (v_hvi < u_hvi) v_ptr++;
                else {
                    res.push_back(u_hvi);
                    u_ptr++;
                    v_ptr++;
                }
            }
        }
    }
}

void HVI::get_common_neis(node u, const hvi_vector &hvis, hvi_vector &res) const {
    vector<HybridVertexInterval> tmp_res({});
    if (hvis.empty()) {
        res = hvi_vector(tmp_res);
        return;
    }
    size_t u_ptr = hvi_offsets[u], v_ptr = 0;
    while (u_ptr < hvi_offsets[u+1] && v_ptr < hvis.size()) {
        HybridVertexInterval u_hvi = hvi_list[u_ptr], v_hvi = hvis.hvi_list[v_ptr];
        if (u_hvi & HVI_RIGHT_BOUNDARY_MASK) {
            u_ptr++;
            continue;
        }
        if (v_hvi & HVI_RIGHT_BOUNDARY_MASK) {
            v_ptr++;
            continue;
        }
        if (u_hvi & HVI_LEFT_BOUNDARY_MASK) {
            u_hvi ^= HVI_LEFT_BOUNDARY_MASK;
            uint32_t u_right = hvi_list[u_ptr+1] & GETNODE_MASK;
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvis.hvi_list[v_ptr+1] & GETNODE_MASK;
                if (u_right < v_hvi) {
                    u_ptr += 2;
                    continue;
                }
                else if (u_hvi > v_right) {
                    v_ptr += 2;
                    continue;
                }
                else {
                    // res += min(u_right, v_right) - max(u_hvi, v_hvi) + 1;
                    tmp_res.push_back(max(u_hvi, v_hvi) | HVI_LEFT_BOUNDARY_MASK);
                    tmp_res.push_back(min(u_right, v_right) | HVI_RIGHT_BOUNDARY_MASK);
                    if (tmp_res.back() == 0xffffffff) printf("u_right: %u, u_hvi: %u, v_right: %u, v_hvi: %u", u_right, u_hvi, v_right, v_hvi);
                    if (u_right < v_right) u_ptr += 2;
                    else if (u_right > v_right) v_ptr += 2;
                    else {
                        u_ptr += 2;
                        v_ptr += 2;
                    }
                }
            }
            else {
                if (v_hvi < u_hvi) v_ptr++;
                else if (v_hvi > u_right) u_ptr++;
                else {
                    // res++;
                    tmp_res.push_back(v_hvi);
                    v_ptr++;
                }
            }
        }
        else {
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvis.hvi_list[v_ptr+1] & GETNODE_MASK;
                if (u_hvi < v_hvi) u_ptr++;
                else if (u_hvi > v_right) v_ptr += 2;
                else {
                    // res++;
                    tmp_res.push_back(u_hvi);
                    u_ptr++;
                }
            }
            else {
                if (u_hvi < v_hvi) u_ptr++;
                else if (v_hvi < u_hvi) v_ptr++;
                else {
                    // res++;
                    tmp_res.push_back(u_hvi);
                    u_ptr++;
                    v_ptr++;
                }
            }
        }
    }
    res = hvi_vector(tmp_res);
}

void HVI::get_common_neis(node u, const vector<node> &nodes, vector<node> &res) const {
    res.clear();
    res.reserve(nodes.size());
    size_t u_ptr = hvi_offsets[u], v_ptr = 0;
    while (u_ptr < hvi_offsets[u+1] && v_ptr < nodes.size()) {
        if (hvi_list[u_ptr] & HVI_LEFT_BOUNDARY_MASK) {
            node u_left = hvi_list[u_ptr] & GETNODE_MASK;
            node u_right = hvi_list[u_ptr+1] & GETNODE_MASK;
            if (nodes[v_ptr] < u_left) v_ptr++;
            else if (nodes[v_ptr] > u_right) u_ptr += 2;
            else {
                res.push_back(nodes[v_ptr]);
                v_ptr++;
            }
        }
        else {
            node u = hvi_list[u_ptr];
            if (u < nodes[v_ptr]) u_ptr++;
            else if (nodes[v_ptr] < u) v_ptr++;
            else {
                res.push_back(u);
                u_ptr++;
                v_ptr++;
            }
        }
    }
}

size_t HVI::cnt_common_neighbor(node u, node v) const {
    size_t u_ptr = hvi_offsets[u], v_ptr = hvi_offsets[v], res = 0;
    while (u_ptr < hvi_offsets[u+1] && v_ptr < hvi_offsets[v+1]) {
        HybridVertexInterval u_hvi = hvi_list[u_ptr], v_hvi = hvi_list[v_ptr];
        if (u_hvi & HVI_RIGHT_BOUNDARY_MASK) {
            u_ptr++;
            continue;
        }
        if (v_hvi & HVI_RIGHT_BOUNDARY_MASK) {
            v_ptr++;
            continue;
        }
        if (u_hvi & HVI_LEFT_BOUNDARY_MASK) {
            u_hvi ^= HVI_LEFT_BOUNDARY_MASK;
            uint32_t u_right = hvi_list[u_ptr+1] & GETNODE_MASK;
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvi_list[v_ptr+1] & GETNODE_MASK;
                if (u_right < v_hvi) {
                    u_ptr += 2;
                    continue;
                }
                else if (u_hvi > v_right) {
                    v_ptr += 2;
                    continue;
                }
                else {
                    res += min(u_right, v_right) - max(u_hvi, v_hvi) + 1;
                    if (u_right < v_right) u_ptr += 2;
                    else if (u_right > v_right) v_ptr += 2;
                    else {
                        u_ptr += 2;
                        v_ptr += 2;
                    }
                }
            }
            else {
                if (v_hvi < u_hvi) v_ptr++;
                else if (v_hvi > u_right) u_ptr++;
                else {
                    res++;
                    v_ptr++;
                }
            }
        }
        else {
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvi_list[v_ptr+1] & GETNODE_MASK;
                if (u_hvi < v_hvi) u_ptr++;
                else if (u_hvi > v_right) v_ptr += 2;
                else {
                    res++;
                    u_ptr++;
                }
            }
            else {
                if (u_hvi < v_hvi) u_ptr++;
                else if (v_hvi < u_hvi) v_ptr++;
                else {
                    res++;
                    u_ptr++;
                    v_ptr++;
                }
            }
        }
    }
    return res;
}

size_t HVI::cnt_common_neighbor(node u, hvi_vector &hvis) const {
    size_t u_ptr = hvi_offsets[u], v_ptr = 0, res = 0;
    while (u_ptr < hvi_offsets[u+1] && v_ptr < hvis.hvi_list.size()) {
        HybridVertexInterval u_hvi = hvi_list[u_ptr], v_hvi = hvis.hvi_list[v_ptr];
        if (u_hvi & HVI_RIGHT_BOUNDARY_MASK) {
            u_ptr++;
            continue;
        }
        if (v_hvi & HVI_RIGHT_BOUNDARY_MASK) {
            v_ptr++;
            continue;
        }
        if (u_hvi & HVI_LEFT_BOUNDARY_MASK) {
            u_hvi ^= HVI_LEFT_BOUNDARY_MASK;
            uint32_t u_right = hvi_list[u_ptr+1] & GETNODE_MASK;
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvis.hvi_list[v_ptr+1] & GETNODE_MASK;
                if (u_right < v_hvi) {
                    u_ptr += 2;
                    continue;
                }
                else if (u_hvi > v_right) {
                    v_ptr += 2;
                    continue;
                }
                else {
                    res += min(u_right, v_right) - max(u_hvi, v_hvi) + 1;
                    if (u_right < v_right) u_ptr += 2;
                    else if (u_right > v_right) v_ptr += 2;
                    else {
                        u_ptr += 2;
                        v_ptr += 2;
                    }
                }
            }
            else {
                if (v_hvi < u_hvi) v_ptr++;
                else if (v_hvi > u_right) u_ptr++;
                else {
                    res++;
                    v_ptr++;
                }
            }
        }
        else {
            if (v_hvi & HVI_LEFT_BOUNDARY_MASK) {
                v_hvi ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = hvis.hvi_list[v_ptr+1] & GETNODE_MASK;
                if (u_hvi < v_hvi) u_ptr++;
                else if (u_hvi > v_right) v_ptr += 2;
                else {
                    res++;
                    u_ptr++;
                }
            }
            else {
                if (u_hvi < v_hvi) u_ptr++;
                else if (v_hvi < u_hvi) v_ptr++;
                else {
                    res++;
                    u_ptr++;
                    v_ptr++;
                }
            }
        }
    }
    return res;
}

size_t HVI::tc() const {
    size_t res = 0;
    for (node u = 0; u < num_nodes; ++u) {
        node curr_head = 0;
        for (size_t pos = hvi_offsets[u]; pos < hvi_offsets[u+1]; pos++) {
            HybridVertexInterval hvi = hvi_list[pos];
            if (hvi & HVI_LEFT_BOUNDARY_MASK) {
                curr_head = hvi ^ HVI_LEFT_BOUNDARY_MASK;
                continue;
            }
            if (hvi & HVI_RIGHT_BOUNDARY_MASK) {
                node next = curr_head;
                node right = hvi & GETNODE_MASK;
                while (next <= right) {
                    res += cnt_common_neighbor(u, next);
                    next++;
                }    
            }
            else {
                res += cnt_common_neighbor(u, hvi);
            }
        }
    }
    return res;
}

void HVI::page_rank(vector<double> &res, double epsilon, size_t max_iter) {
    res = vector<double>(num_nodes, (double) 1);
	vector<double> tmp_res(num_nodes, 0);
    vector<size_t> degs(num_nodes, 0);
	double diff = (double) num_nodes;
	size_t iter = 0;
    for (size_t u = 0; u < num_nodes; u++) {
        for (size_t i = hvi_offsets[u]; i < hvi_offsets[u+1]; i++) {
            if (hvi_list[i] & HVI_LEFT_BOUNDARY_MASK) {
                degs[u] += (hvi_list[i+1] & GETNODE_MASK) - (hvi_list[i] & GETNODE_MASK);
            }
            else {
                degs[u]++;
            }
        }
    }
	while (diff > epsilon && iter < max_iter) {
    // while (iter < max_iter) {
		iter++;
		diff = 0;
		swap(res, tmp_res);
        for (size_t u = 0; u < num_nodes; u++) {
			tmp_res[u] /= degs[u];
            res[u] = 0;
		}
		for (size_t u = 0; u < num_nodes; u++) {
            if (hvi_offsets[u] == hvi_offsets[u+1]) continue;
            for (hvi_iterator iter(*this, u); iter.curr_pos != iter.end_pos; ++iter) {
                node v = iter.curr_val;
                res[u] += tmp_res[v];
                // res[v] += tmp_res[u];
            }
            diff += fabs(res[u] - tmp_res[u] * degs[u]);
		}
	}
    printf("-------- PageRank value --------\n");
	for (size_t u = 0; u < 10; u++) {
		printf("Node %lu, pr value: %.3lf.\n", u, res[u]);
	}
} 

void HVI::subgraph_matching_helper(const Graph &q, unordered_set<node> &partial_matching, vector<int> &join_order, vector<hvi_vector> &inter_res, ull &res) {
    vector<node> neis;
    vector<HybridVertexInterval> hvis;
    node curr_join_idx = partial_matching.size();
    // printf("Current partial matching size: %d.\n", curr_join_idx);
    q.get_neis(curr_join_idx, neis);
    vector<node> join_idx_sequence;
    join_idx_sequence.reserve(neis.size());
    for (node v : neis) {
        if (join_order[v] < join_order[curr_join_idx]) join_idx_sequence.emplace_back(v);
    }
    if (join_idx_sequence.empty()) return;
    // printf("Check point! Get join sqeuence.\n");
    
    get_neis(inter_res[join_idx_sequence[0]].back(), hvis);
    hvi_vector tmp(hvis);
    vector<HybridVertexInterval>().swap(hvis);
    vector<node>().swap(neis);
    // printf("Check point! Get neis.\n");

    for (auto join_idx : join_idx_sequence) {
        if (join_idx == join_idx_sequence[0]) continue;
        get_common_neis(inter_res[join_idx].back(), tmp, inter_res[curr_join_idx]);
        swap(tmp, inter_res[curr_join_idx]);
    }
    swap(tmp, inter_res[curr_join_idx]);
    
    if (curr_join_idx == join_order.size() - 1) {
        res += inter_res[curr_join_idx].get_decoded_size(partial_matching);
        return;
    }
    
    while (!inter_res[curr_join_idx].empty()) {
        node v = inter_res[curr_join_idx].back();
        if (!partial_matching.count(v)) {
            partial_matching.insert(v);
            subgraph_matching_helper(q, partial_matching, join_order, inter_res, res);
            partial_matching.erase(v);
        }
        inter_res[curr_join_idx].pop();
    }
    return;
}

void HVI::subgraph_matching_helper(const Graph &q, vector<node> &partial_matching, vector<int> &join_order, const vector<size_t> &degs, ull &res) {
    vector<node> neis;
    vector<HybridVertexInterval> hvis;
    node curr_join_idx = partial_matching.size();
    // printf("Current partial matching size: %d.\n", curr_join_idx);
    q.get_neis(curr_join_idx, neis);
    vector<node> join_idx_sequence;
    join_idx_sequence.reserve(neis.size());
    for (node v : neis) {
        if (join_order[v] < join_order[curr_join_idx]) join_idx_sequence.emplace_back(v);
    }
    if (join_idx_sequence.empty()) return;
    // printf("Check point! Get join sqeuence.\n");

    node min_hvi_vertex = partial_matching[join_idx_sequence[0]];
    int min_hvi = hvi_offsets[min_hvi_vertex+1] - hvi_offsets[min_hvi_vertex];
    for (int v : join_idx_sequence) {
        int n_hvi = hvi_offsets[partial_matching[v]+1] - hvi_offsets[partial_matching[v]];
        if (n_hvi < min_hvi) {
            min_hvi = n_hvi;
            min_hvi_vertex = partial_matching[v];
        }
    }
    // printf("Check point! Get min hvi.\n");
    get_neis(min_hvi_vertex, hvis);
    hvi_vector tmp(hvis);
    hvi_vector candidates;
    vector<node>().swap(neis);
    // printf("Check point! Get neis.\n");

    if (join_idx_sequence.size() == 1) {
        swap(tmp, candidates);
    }
    else {
        if (curr_join_idx == join_order.size() - 1) {
            node end_join_idx = (min_hvi_vertex == partial_matching[join_idx_sequence.back()]) ? join_idx_sequence[join_idx_sequence.size()-2] : join_idx_sequence.back();
            for (auto join_idx : join_idx_sequence) {
                if (partial_matching[join_idx] == min_hvi_vertex) continue;
                if (join_idx == end_join_idx) {
                    res += cnt_common_neighbor(partial_matching[join_idx], tmp);
                    for (node u : partial_matching) {
                        if (tmp.has_element(u) && has_directed_edge(partial_matching[join_idx], u))
                            res--;
                    }
                    // get_common_neis(partial_matching[join_idx], tmp, candidates);
                    // res += candidates.get_decoded_size(partial_matching);
                    return;
                }
                else {
                    get_common_neis(partial_matching[join_idx], tmp, candidates);
                    swap(tmp, candidates);
                }    
            }
        }
        else {
            for (auto join_idx : join_idx_sequence) {
                if (partial_matching[join_idx] == min_hvi_vertex) continue;
                get_common_neis(partial_matching[join_idx], tmp, candidates);
                swap(tmp, candidates);
            }
            swap(tmp, candidates);
        }
    }
    
    // for (auto join_idx : join_idx_sequence) {
    //     if (partial_matching[join_idx] == min_hvi_vertex) continue;
    //     get_common_neis(partial_matching[join_idx], tmp, candidates);
    //     swap(tmp, candidates);
    // }
    // swap(tmp, candidates);
    
    // if (curr_join_idx == join_order.size() - 1) {
    //     res += candidates.get_decoded_size(partial_matching);
    //     return;
    // }
    
    // unordered_set<node> hash_partial_matching(partial_matching.begin(), partial_matching.end());
    while (!candidates.empty()) {
        node v = candidates.back();
        candidates.pop();
        if (degs[v] < q.get_deg(curr_join_idx)) continue;
        bool is_duplicate = false;
        for (node u : partial_matching) {
            if (u == v) {
                is_duplicate = true;
                break;
            }
        }
        if (is_duplicate) continue;
        // if (hash_partial_matching.count(v)) continue;
        partial_matching.push_back(v);
        subgraph_matching_helper(q, partial_matching, join_order, degs, res);
        partial_matching.pop_back(); 
    }
    return;
}