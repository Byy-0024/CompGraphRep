#include "CompressGraph.hpp"

using namespace std;

CompressGraphIterator::CompressGraphIterator(const CompressGraph &g, node u) {
    csr_vlist = g.get_csr_vlist();
    csr_elist = g.get_csr_elist();
    curr_ptr = csr_vlist[u];
    end_ptr = csr_vlist[u+1];
    num_nodes = g.get_num_nodes();
    node v = csr_elist[curr_ptr];
    buffer = {};
    buffer_ptr = 0;
    if (v >= num_nodes) {
        g.decode_rule(v, buffer);
    }
}

CompressGraphIterator& CompressGraphIterator::operator++() {
    if (!buffer.empty() && buffer_ptr < buffer.size() - 1) {
        buffer_ptr++;
    }
    else {
        curr_ptr++;
        buffer.clear();
        if (curr_ptr < end_ptr) {
            node val = csr_elist[curr_ptr];
            if (val >= num_nodes) {
                decode_rule(val);
                buffer_ptr = 0;
            }
        }
    }
    return *this;
}

void CompressGraphIterator::decode_rule(node r) {
    for (int i = csr_vlist[r]; i < csr_vlist[r+1]; i++) {
        node v = csr_elist[i];
        if (v < num_nodes) buffer.push_back(v);
        else {
            decode_rule(v);
        }
    }
}

node CompressGraphIterator::get_val() {
    // printf("curr_ptr: %lu, end_ptr %lu, curr_val: %d, rule_left: %d, rule_right: %d\n", curr_ptr, end_ptr, curr_val, rule_left, rule_right);
    return buffer.empty() ? csr_elist[curr_ptr] : buffer[buffer_ptr];
}

CompressGraph::CompressGraph(const std::string csr_vlist_binfile, const std::string csr_elist_binfile, const size_t _num_nodes) {
    num_nodes = _num_nodes;
    ifstream input_csr_vlist(csr_vlist_binfile, ios::binary);
    if (!input_csr_vlist.is_open()) {
        cout << "Unable to open csr vlist binfile" << endl;
        return;
    }
    input_csr_vlist.seekg(0, ios::end);
    streamsize size = input_csr_vlist.tellg();
    input_csr_vlist.seekg(0, ios::beg);
    csr_vlist.resize(size / sizeof(int));
    input_csr_vlist.read(reinterpret_cast<char*>(csr_vlist.data()), size);
    input_csr_vlist.close();
    num_rules = csr_vlist.size() - num_nodes;

    ifstream input_csr_elist(csr_elist_binfile, ios::binary);
    if (!input_csr_elist.is_open()) {
        cout << "Unable to open csr elist binfile" << endl;
        return;
    }
    input_csr_elist.seekg(0, ios::end);
    size = input_csr_elist.tellg();
    input_csr_elist.seekg(0, ios::beg);
    csr_elist.resize(size / sizeof(node));
    input_csr_elist.read(reinterpret_cast<char*>(csr_elist.data()), size);
    input_csr_elist.close();
    num_edges = csr_elist.size();
    csr_vlist.push_back(num_edges);

    printf("Has read compress graph! #Vertices: %lu, #Rules: %lu.\n", num_nodes, num_rules);
}

const int* CompressGraph::get_csr_vlist() const {
    return csr_vlist.data();
}

const node* CompressGraph::get_csr_elist() const {
    return csr_elist.data();
}

bool CompressGraph::has_directed_edge(node u, node v) const {
    // CompressGraphIterator it(*this, u);
    // while (it.curr_ptr != it.end_ptr) {
    //     node curr_val = it.get_val();
    //     if (curr_val == v) return true;
    //     if (curr_val > v) return false;
    //     ++it;
    // }
    // return false;

    int __first = csr_vlist[u], __last = csr_vlist[u+1];
    int __len = __last - __first;
    while (__len > 0) {
        int __half = __len >> 1;
        int __middle = __first + __half;
        if (csr_elist[__middle] == v) return true;
        if (csr_elist[__middle] < v){
            __first = __middle;
            ++__first;
            __len = __len - __half - 1;
        }
        else if (csr_elist[__middle] < num_nodes) __len = __half;
        else {
            node left = get_rule_min(csr_elist[__middle]), right = get_rule_max(csr_elist[__middle]);
            if (right < v) {
                __first = __middle;
                ++__first;
                __len = __len - __half - 1;
            }
            else if (left > v) __len = __half;
            else {
                return cmp_node_with_rule(v, csr_elist[__middle]);
            }
        }
    }
    return csr_elist[__first] == v && __first != __last;

    // vector<node> buffer;
    // decode_rule(u, buffer);
    // size_t lower_bound_pos = lower_bound(buffer.begin(), buffer.end(), v) - buffer.begin();
    // return (lower_bound_pos != buffer.size() && buffer[lower_bound_pos] == v);
}

bool CompressGraph::has_edge(node u, node v) const {
    return (csr_vlist[u+1] - csr_vlist[u] < csr_vlist[v+1] - csr_vlist[v]) ? has_directed_edge(u, v) : has_directed_edge(v, u);
}

// Return value   &&   Meaning
//      0         &&   rule r covers but does not contain the vertex u 
//      1         &&   rule r contains the vertex u
//   INT_MIN      &&   all vertices in rule r are smaller than u
//   INT_MAX      &&   all vertices in rule r are larger than u
// int CompressGraph::cmp_node_with_rule (node u, node r) const {
//     node left = csr_elist[csr_vlist[r]], right = csr_elist[csr_vlist[r]+1];
//     if (left == u || right == u) return 1;
//     if (right < u) return INT_MIN;
//     if (left < num_nodes) {
//         if (left > u) return INT_MAX;
//         if (right < num_nodes) return 0;
//         int stat = cmp_node_with_rule(u, right);
//         return (stat == INT_MAX) ? 0 : stat;
//     }
//     else {
//         int stat = cmp_node_with_rule(u, left);
//         if (stat == INT_MIN) {
//             if (right < num_nodes) return 0;
//             stat = cmp_node_with_rule(u, right);
//             return (stat == INT_MAX) ? 0 : stat;
//         }
//         return stat;
//     }
// }

bool CompressGraph::cmp_node_with_rule (node u, node r) const {
    vector<node> buffer;
    decode_rule(r, buffer);
    size_t lower_bound_pos = lower_bound(buffer.begin(), buffer.end(), u) - buffer.begin();
    return (lower_bound_pos != buffer.size() && buffer[lower_bound_pos] == u);
}

node CompressGraph::get_rule_min(node r) const {
    node left = csr_elist[csr_vlist[r]];
    return (left < num_nodes) ? left : get_rule_min(left);
    // vector<node> buffer;
    // decode_rule(r, buffer);
    // return buffer[0];
}

node CompressGraph::get_rule_max(node r) const {
    node right = csr_elist[csr_vlist[r+1]-1];
    return (right < num_nodes) ? right : get_rule_max(right);
    // vector<node> buffer;
    // decode_rule(r, buffer);
    // return buffer[buffer.size()-1];
}

size_t CompressGraph::cnt_common_neis(node u, node v) const {
    CompressGraphIterator u_iter(*this, u), v_iter(*this, v);
    node u_nei = u_iter.get_val(), v_nei = v_iter.get_val();
    size_t res = 0;
    while (u_iter.curr_ptr < u_iter.end_ptr && v_iter.curr_ptr < v_iter.end_ptr) {
        if (u_nei == v_nei) {
            res++;
            ++u_iter;
            u_nei = u_iter.get_val();
            ++v_iter;
            v_nei = v_iter.get_val();
        }
        else if (u_nei < v_nei) {
            ++u_iter;
            u_nei = u_iter.get_val();
        }
        else {
            ++v_iter;
            v_nei = v_iter.get_val();
        }
    }
    return res;
}

size_t CompressGraph::cnt_common_neis(const vector<node> &neis1, const vector<node> &neis2) const {
    size_t u_ptr = 0, v_ptr = 0, res = 0;
    while (u_ptr < neis1.size() && v_ptr < neis2.size()) {
        if (neis1[u_ptr] == neis2[v_ptr]) {
            res++;
            u_ptr++;
            v_ptr++;
        }
        else if (neis1[u_ptr] < neis2[v_ptr]) u_ptr++;
        else v_ptr++;
    }
    return res;
}

size_t CompressGraph::get_deg(node u) const {
    vector<node> neis;
    get_neis(u, neis);
    size_t res = neis.size();
    vector<node>().swap(neis);
    return res;
}

size_t CompressGraph::get_num_edges() const {
    return num_edges;
}

size_t CompressGraph::get_num_nodes() const {
    return num_nodes;
}

size_t CompressGraph::get_num_rules() const {
    return num_rules;
}

size_t CompressGraph::size_in_bytes() const {
    return csr_vlist.size() * sizeof(int) + csr_elist.size() * sizeof(node);
}

void CompressGraph::decode_rule(node r, vector<node> &neis) const {
    for (int i = csr_vlist[r]; i < csr_vlist[r+1]; i++) {
        node v = csr_elist[i];
        if (v < num_nodes) neis.push_back(v);
        else {
            decode_rule(v, neis);
        }
    }
}

void CompressGraph::get_common_neis(const vector<node> &neis1, const vector<node> &neis2, vector<node> &res) const {
    size_t u_ptr = 0, v_ptr = 0;
    while (u_ptr < neis1.size() && v_ptr < neis2.size()) {
        if (neis1[u_ptr] == neis2[v_ptr]) {
            res.push_back(neis1[u_ptr]);
            u_ptr++;
            v_ptr++;
        }
        else if (neis1[u_ptr] < neis2[v_ptr]) u_ptr++;
        else v_ptr++;
    }
}

void CompressGraph::get_neis(node u, vector<node> &neis) const {
    neis.clear();
    neis.reserve(csr_vlist[u+1] - csr_vlist[u]);
    for (int i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
        node v = csr_elist[i];
        if (v < num_nodes) neis.push_back(v);
        else {
            decode_rule(v, neis);
        }
    }
    // sort(neis.begin(), neis.end());
}

void CompressGraph::print_neis(node u) const {
    vector<node> neis;
    get_neis(u, neis);
    printf("(After expansion)\nN(%d):", u);
    for (node v : neis) {
        printf(" %d", v);
    }
    printf(".\n");
}


// size_t CompressGraph::tc() {
//     vector<node> neis1, neis2;
//     size_t res = 0;
//     for (node u = 0; u < num_nodes; u++) {
//         get_neis(u, neis1);
//         for (node v : neis1) {
//             get_neis(v, neis2);
//             res += cnt_common_neis(neis1, neis2);
//         }
//     }
//     return res;
// }

size_t CompressGraph::tc() {
    vector<vector<node>> neis(num_nodes);
    size_t res = 0;
    for (node u = 0; u < num_nodes; u++) {
        get_neis(u, neis[u]);
    }
    for (node u = 0; u < num_nodes; u++) {
        for (node v : neis[u]) {
            res += cnt_common_neis(neis[u], neis[v]);
        }
    }
    return res;
}


ull CompressGraph::mc(node u) {
	ull res = 0;
	vector<node> P;
	get_neis(u, P);
	vector<node> X;
	X.reserve(P.size());
	BKP(P, X, res);
	return res;
}

ull CompressGraph::subgraph_matching(const Graph &q) {
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

    vector<vector<node>> inter_res(n);
    unordered_set<node> partial_matching;
    for (node u = 0; u < num_nodes; u++) {
        if (get_deg(u) < q_max_deg) continue;
        inter_res[0].push_back(u);
        partial_matching.insert(u);
        subgraph_matching_helper(q, partial_matching, join_order, inter_res, res);
        inter_res[0].pop_back();
        partial_matching.clear();
    }
    return res;
}

void CompressGraph::BKP(vector<node> &P, vector<node> &X, ull &res) {
	if (P.empty()) {
		if (X.empty()) res++;
		return;
	}
    // printf("BKP start!\n");
	node u = P[0];

    vector<node> NU;
    get_neis(u, NU);
	int vit1 = 0;
	int vit2 = 0;
    int ud = NU.size();
	int ps = P.size();
	int xs = X.size();

	while (vit1 != ps) {
		if (vit2 == ud || P[vit1] < NU[vit2]) {
			int v = P[vit1];
            vector<node> NV;
            get_neis(v, NV);
			vector<node> NP;
			NP.reserve(ps);
			vector<node> NX;
			NX.reserve(xs);
			get_common_neis(NV, X, NX);
			get_common_neis(NV, P, NP);
			BKP(NP, NX, res);
			P.erase(P.begin()+vit1);
			X.insert(lower_bound(X.begin(), X.end(), v), v);
			xs++;
			ps--;
            vector<node>().swap(NV);
            vector<node>().swap(NX);
            vector<node>().swap(NP);
		}
		else if (P[vit1] == NU[vit2]) {
            vit1++;
            vit2++;
        }
        else vit2++;
	}
}

void CompressGraph::bfs(node u, std::vector<size_t> &bfs_rank) {
    bfs_rank.clear();
    bfs_rank.resize(num_nodes, UINT_MAX);
    bfs_rank[u] = 0;

    vector<node> frontiers;
    frontiers.reserve(num_nodes);
    frontiers.push_back(u);

    queue<node> rule_frontiers;
    vector<size_t> rule_bfs_rank(num_rules, UINT_MAX);

    size_t curr_seen_cnt = 0;
    size_t next_seen_cnt = frontiers.size();
    size_t rank = 1;
    while (next_seen_cnt > curr_seen_cnt) {
        for (size_t i = curr_seen_cnt; i < next_seen_cnt; i++) {
            for (size_t j = csr_vlist[frontiers[i]]; j < csr_vlist[frontiers[i] + 1]; j++) {
                node v = csr_elist[j];
                if (v < num_nodes) {
                    if (bfs_rank[v] == UINT_MAX) {
                        bfs_rank[v] = rank;
                        frontiers.push_back(v);
                    }
                }
                else {
                    if (rule_bfs_rank[v - num_nodes] == UINT_MAX) {
                        rule_bfs_rank[v - num_nodes] = rank;
                        rule_frontiers.push(v);
                    }
                }
            }
        }
        while (!rule_frontiers.empty()) {
            node r = rule_frontiers.front();
            rule_frontiers.pop();
            for (size_t j = csr_vlist[r]; j < csr_vlist[r + 1]; j++) {
                node v = csr_elist[j];
                if (v < num_nodes) {
                    if (bfs_rank[v] == UINT_MAX) {
                        bfs_rank[v] = rank;
                        frontiers.push_back(v);
                    }
                }
                else {
                    if (rule_bfs_rank[v - num_nodes] == UINT_MAX) {
                        rule_bfs_rank[v - num_nodes] = rank;
                        rule_frontiers.push(v);
                    }
                }
            } 
        }
        curr_seen_cnt = next_seen_cnt;
        next_seen_cnt = frontiers.size();
        rank++;
    }
}

void CompressGraph::cc(vector<node> &component_id) {
    component_id.clear();
    component_id.resize(num_nodes + num_rules, INT_MAX);

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

void CompressGraph::core_decomposition(vector<size_t> &core_numbers) {
    core_numbers.clear();
    core_numbers.resize(num_nodes);
    vector<pair<node, size_t>> degs;
    vector<node> neis;
    degs.reserve(num_edges);
    for (node u = 0; u < num_nodes; u++) {
        get_neis(u, neis);
        core_numbers[u] = neis.size();
        degs.push_back({u, core_numbers[u]});
    }

    make_heap(degs.begin(), degs.end(), weight_greater<size_t>);

    while (!degs.empty()) {
        pop_heap(degs.begin(), degs.end(), weight_greater<size_t>);
        node u = degs.back().first;
        size_t _core_number = degs.back().second;
        degs.pop_back();
        if (core_numbers[u] < _core_number) continue;
        get_neis(u, neis);
        for (node v : neis) {
            if (core_numbers[v] <= core_numbers[u]) continue;
            core_numbers[v]--;
            degs.push_back({v, core_numbers[v]});
            push_heap(degs.begin(), degs.end(), weight_greater<size_t>);
        }
    }
}

void CompressGraph::dfs_helper(node u, vector<bool> &visited, vector<int> &new2origin) const {
    visited[u] = true;
    new2origin.emplace_back(u);
    vector<node> neis;
    get_neis(u, neis);
    for (node v: neis) {
        if (visited[v] == false) dfs_helper(v, visited, new2origin);
    }
}

void CompressGraph::get_dfs_order(node u, vector<int> &new2origin) const {
    vector<bool> visited(num_nodes, false);
    new2origin.reserve(num_nodes);
    dfs_helper(u, visited, new2origin);
}

void CompressGraph::page_rank(vector<double> &res, double epsilon, size_t max_iter) {
	res = vector<double>(num_nodes, (double) 1);
	vector<double> tmp_res(num_nodes, 0);
    vector<double> tmp_rule_res(num_rules, 0);
	double diff = (double) num_nodes;
	size_t iter = 0;
    vector<size_t> degs(num_nodes, 0);
    vector<node> neis;
    for (node u = 0; u < num_nodes; u++) {
        get_neis(u, neis);
        degs[u] = neis.size();
    }
	while (diff > epsilon && iter < max_iter) {
		iter++;
		diff = 0;
		swap(res, tmp_res);
		for (size_t u = 0; u < num_nodes; u++) {
			tmp_res[u] /= degs[u];
            res[u] = 0;
		}
        tmp_rule_res.clear();
        tmp_rule_res.resize(num_rules, 0);

        for (size_t u = 0; u < num_rules; u++) {
            for (size_t i = csr_vlist[u + num_nodes]; i < csr_vlist[u + num_nodes + 1]; i++) {
                node v = csr_elist[i];
                if (v < num_nodes) tmp_rule_res[u] += tmp_res[v];
                else tmp_rule_res[u] += tmp_rule_res[v - num_nodes];
			}
        }

		for (size_t u = 0; u < num_nodes; u++) {
            if (csr_vlist[u] == csr_vlist[u+1]) continue;
			for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
                node v = csr_elist[i];
                if (v < num_nodes) res[u] += tmp_res[v];
                else res[u] += tmp_rule_res[v - num_nodes];
			}
            diff += fabs(res[u] - tmp_res[u] * degs[u]);
		}
	}
	printf("-------- PageRank value --------\n");
	for (size_t u = 0; u < 10; u++) {
		printf("Node %lu, pr value: %.3lf.\n", u, res[u]);
	}
}

void CompressGraph::subgraph_matching_helper(const Graph &q, unordered_set<node> &partial_matching, vector<int> &join_order, vector<vector<node>> &inter_res, ull &res) {
    vector<node> neis1, neis2;
    node curr_join_idx = partial_matching.size();
    q.get_neis(curr_join_idx, neis1);
    vector<node> join_idx_sequence;
    join_idx_sequence.reserve(neis1.size());
    for (node v : neis1) {
        if (join_order[v] < join_order[curr_join_idx]) join_idx_sequence.emplace_back(v);
    }
    if (join_idx_sequence.empty()) return;
    get_neis(inter_res[join_idx_sequence[0]].back(), neis1);
    for (auto join_idx : join_idx_sequence) {
        if (join_idx == join_idx_sequence[0]) continue;
        get_neis(inter_res[join_idx].back(), neis2);
        get_common_neis(neis1, neis2, inter_res[curr_join_idx]);
        swap(neis1, inter_res[curr_join_idx]);
    }
    inter_res[curr_join_idx].clear();
    for (auto v : neis1) {
        if (partial_matching.count(v)) continue;
        inter_res[curr_join_idx].push_back(v);
    }
    vector<node>().swap(neis1);
    vector<node>().swap(neis2);

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