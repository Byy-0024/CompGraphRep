#include "Interval.hpp"

bool hybrid_vid_less(const HybridVertexInterval &x, const HybridVertexInterval &y) {
    return (x & GETNODE_MASK) < (y & GETNODE_MASK);
}

bool hybrid_vid_greater(const HybridVertexInterval &x, const HybridVertexInterval &y) {
    return (x & GETNODE_MASK) > (y & GETNODE_MASK);
}

bool bfs_fast_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank){
    for (int i = 0; i < bfs_rank.size(); ++i) {
        if (bfs_rank[i] != gt_bfs_rank[i]) return false;
    }
    return true;
}

bool bfs_fast_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank, vector<node> &origin_to_new){
    for (int i = 0; i < bfs_rank.size(); ++i) {
        if (bfs_rank[origin_to_new[i]] != gt_bfs_rank[i]) return false;
    }
    return true;
}

void bfs_full_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank){
    size_t error_cnt = 0;
    for (int i = 0; i < bfs_rank.size(); ++i) {
        if (bfs_rank[i] != gt_bfs_rank[i]) {
            printf("Error at origin id: %u, groudtruth rank: %d, rank: %d!\n", i, gt_bfs_rank[i], bfs_rank[i]);
            error_cnt++;
            if (error_cnt > 20) break;
        }
    }
}

void bfs_full_verification(vector<int> &bfs_rank, vector<int> &gt_bfs_rank, vector<node> &origin_to_new){
    for (int i = 0; i < bfs_rank.size(); ++i) {
        if (bfs_rank[origin_to_new[i]] != gt_bfs_rank[i]) {
            printf("Error at origin id: %u, groudtruth rank: %d, new id: %lu, rank: %d!\n", i, gt_bfs_rank[i], origin_to_new[i], bfs_rank[origin_to_new[i]]);
        }
    }
}

UI::UI(const Graph &g) {
    num_nodes = g.get_num_nodes();
    num_edges = g.get_num_edges();
    inds = (size_t *)malloc((num_nodes+1) * sizeof(size_t));
    vector<Interval> tmp_intervals;
    tmp_intervals.reserve(num_nodes);
    vector<node> neis;
    for (node u = 0; u < num_nodes; ++u) {
        g.get_neighbors(u, neis);
        inds[u] = tmp_intervals.size();
        node left = neis[0];
        node right = left;
        for (auto v : neis) {
            if (v == right) {
                right++;
            }
            else {
                tmp_intervals.push_back((Interval) left << 32 + right);
                left = v;
                right = v+1;
            }
        }
        tmp_intervals.push_back((Interval) left << 32 + right);
    }
    inds[num_nodes] = tmp_intervals.size();
    intervals = (Interval*)malloc(inds[num_nodes] * sizeof(Interval));
    memcpy(intervals, tmp_intervals.data(), inds[num_nodes] * sizeof(Interval));
}

size_t UI::size_in_bytes() const {
    return inds[num_nodes] * sizeof(Interval);
}

HVI::HVI(const Graph &g) {
    num_nodes = g.get_num_nodes();
    num_hybrid_vertex_intervals = 0;
    inds = (size_t*)malloc(sizeof(size_t)*(num_nodes + 1));
    vector<HybridVertexInterval> tmp_vals;
    tmp_vals.reserve(num_nodes);
    size_t inds_tail_pos = 0;
    vector<node> neis;
    for (size_t i = 0; i < num_nodes; ++i) {
        inds[inds_tail_pos++]=tmp_vals.size();
        g.get_neighbors(i, neis);
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
                if (curr_length == 1) tmp_vals.push_back(curr_head);
                else if (curr_length == 2) {
                    tmp_vals.push_back(curr_head);
                    tmp_vals.push_back(expect_node-1);
                }
                else {
                    tmp_vals.push_back(curr_head | HVI_LEFT_BOUNDARY_MASK);
                    tmp_vals.push_back((expect_node - 1) | HVI_RIGHT_BOUNDARY_MASK);
                }
                curr_head = v;
                expect_node = v+1;
                curr_length = 1;
            }
        }
        if (curr_length == 1) tmp_vals.push_back(curr_head);
        else {
            tmp_vals.push_back(curr_head | HVI_LEFT_BOUNDARY_MASK);
            tmp_vals.push_back((expect_node - 1) | HVI_RIGHT_BOUNDARY_MASK);
        }
    }
    inds[inds_tail_pos++]=tmp_vals.size();
    num_hybrid_vertex_intervals = tmp_vals.size();
    vals = (HybridVertexInterval*)malloc(sizeof(HybridVertexInterval) * num_hybrid_vertex_intervals);
    memcpy(vals, tmp_vals.data(), sizeof(HybridVertexInterval) * inds[num_nodes]);
}

// return if node u is contained in the node v's neighbors.
bool HVI::bsearch(node u, node v, size_t pos) const {
    if (inds[v] == inds[v+1] || (vals[inds[v]] & GETNODE_MASK) > u) {
		pos = 0;
		return false;
	}
    if (vals[inds[v+1]-1] & GETNODE_MASK < u) {
		pos = inds[v+1] - 1;
		return false;
	}
	int left = inds[v], right = inds[v+1] - 1;
	while (right - left > 1) {
		int tmp = (left + right) >> 1;
		if (vals[tmp] & GETNODE_MASK == u) {
			pos = tmp;
			return true;
		}
		else if (vals[tmp] & GETNODE_MASK < u) left = tmp;
		else right = tmp;
	}
	pos = left;
	return vals[pos] & HVI_LEFT_BOUNDARY_MASK;
}

bool HVI::has_directed_edge(const node u, const node v) const {
    if (inds[u+1] == inds[u]) return false;
    size_t lower_bound_pos = lower_bound(vals+inds[u], vals+inds[u+1], v, hybrid_vid_less) - vals;
    if (lower_bound_pos == inds[u+1]) return false;
    if (vals[lower_bound_pos] & HVI_RIGHT_BOUNDARY_MASK) {
        return true;
    }
    else {
        return (vals[lower_bound_pos] & GETNODE_MASK) == v;
    }
}

bool HVI::has_edge(node u, node v) const {
    if (inds[u] == inds[u+1] || inds[v] == inds[v+1]) return false;
    if (inds[u+1] - inds[u] < inds[v+1] - inds[v]) {
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
    return num_hybrid_vertex_intervals;
}

size_t HVI::size_in_bytes() const {
    return num_hybrid_vertex_intervals * sizeof(HybridVertexInterval);
}

size_t* HVI::get_inds() const {
    return inds;
}

HybridVertexInterval* HVI::get_vals() const {
    return vals;
}


void HVI::prefetch_neighbor(node u) {
    size_t prefetch_step = 64 / sizeof(HybridVertexInterval);
    for (size_t pos = inds[u]; pos < inds[u+1]; pos += prefetch_step) {
        _mm_prefetch(&vals[pos], _MM_HINT_T2);
    }
}

void HVI::print_neighbor(node u) const {
    for (size_t pos = inds[u]; pos < inds[u+1]; pos++) {
        HybridVertexInterval hrnode = vals[pos];
        if (hrnode & HVI_LEFT_BOUNDARY_MASK) {
            printf(" [%u,", hrnode & GETNODE_MASK);
        }
        else if (hrnode & HVI_RIGHT_BOUNDARY_MASK) {
            printf(" %u],", hrnode & GETNODE_MASK);
        }
        else printf(" %u ", hrnode);
    }
    printf("\n");
}

void HVI::bfs(node u, vector<int> &bfs_rank) {
    vector<node> frontier(num_nodes, UINT_MAX);
    size_t tail_pos = 0;
    node curr = u;
    frontier[tail_pos++] = curr;
    int rank = 1;
    bfs_rank[curr] = 0;
    size_t curr_seen_cnt = 0;
    size_t next_seen_cnt = tail_pos;
    while (next_seen_cnt > curr_seen_cnt) {
        for (size_t i = curr_seen_cnt; i < next_seen_cnt; i++) {
            curr = frontier[i];
            for (size_t curr_pos = inds[curr]; curr_pos < inds[curr+1];) {
                HybridVertexInterval hrnode = vals[curr_pos];
                if (hrnode & HVI_LEFT_BOUNDARY_MASK) {
                    node next =  hrnode & GETNODE_MASK;
                    node right = vals[curr_pos+1] & GETNODE_MASK;
                    while (next <= right) {
                        if (bfs_rank[next] == INT_MAX) {
                            bfs_rank[next] = rank;
                            frontier[tail_pos++] = next;
                        }
                        next++;
                    }
                    curr_pos += 2;
                }
                else {
                    if (bfs_rank[hrnode] == INT_MAX) {
                        frontier[tail_pos++] = hrnode;
                        bfs_rank[hrnode] = rank;
                    }
                    curr_pos ++;
                }
            }
        } 
        curr_seen_cnt = next_seen_cnt;
        next_seen_cnt = tail_pos;
        rank++;
    }
}

void HVI::get_common_neighbor(node u, node v, vector<HybridVertexInterval> &res) const {
    res.clear();
    size_t u_ptr = inds[u], v_ptr = inds[v];
    while (u_ptr < inds[u+1] && v_ptr < inds[v+1]) {
        HybridVertexInterval u_hrnode = vals[u_ptr], v_hrnode = vals[v_ptr];
        if (u_hrnode & HVI_RIGHT_BOUNDARY_MASK) {
            u_ptr++;
            continue;
        }
        if (v_hrnode & HVI_RIGHT_BOUNDARY_MASK) {
            v_ptr++;
            continue;
        }
        if (u_hrnode & HVI_LEFT_BOUNDARY_MASK) {
            u_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
            uint32_t u_right = vals[u_ptr+1] & GETNODE_MASK;
            if (v_hrnode & HVI_LEFT_BOUNDARY_MASK) {
                v_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = vals[v_hrnode+1] & HVI_RIGHT_BOUNDARY_MASK;
                if (u_right < v_hrnode) {
                    u_ptr += 2;
                    continue;
                }
                else if (u_hrnode > v_right) {
                    v_ptr += 2;
                    continue;
                }
                else {
                    res.push_back(max(u_hrnode, v_hrnode) | HVI_LEFT_BOUNDARY_MASK);
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
                if (v_hrnode < u_hrnode) v_ptr++;
                else if (v_hrnode >= u_right) u_ptr++;
                else {
                    res.push_back(v_hrnode);
                    v_ptr++;
                }
            }
        }
        else {
            if (v_hrnode & HVI_LEFT_BOUNDARY_MASK) {
                v_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = vals[v_hrnode+1] & GETNODE_MASK;
                if (u_hrnode < v_hrnode) u_ptr++;
                else if (u_hrnode > v_right) v_ptr++;
                else {
                    res.push_back(u_hrnode);
                    u_ptr++;
                }
            }
            else {
                if (u_hrnode < v_hrnode) u_ptr++;
                else if (v_hrnode < u_hrnode) v_ptr++;
                else {
                    res.push_back(u_hrnode);
                    u_ptr++;
                    v_ptr++;
                }
            }
        }
    }
}

size_t HVI::cnt_common_neighbor(node u, node v) const {
    size_t u_ptr = inds[u], v_ptr = inds[v], res = 0;
    while (u_ptr < inds[u+1] && v_ptr < inds[v+1]) {
        HybridVertexInterval u_hrnode = vals[u_ptr], v_hrnode = vals[v_ptr];
        if (u_hrnode & HVI_RIGHT_BOUNDARY_MASK) {
            u_ptr++;
            continue;
        }
        if (v_hrnode & HVI_RIGHT_BOUNDARY_MASK) {
            v_ptr++;
            continue;
        }
        if (u_hrnode & HVI_LEFT_BOUNDARY_MASK) {
            u_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
            uint32_t u_right = vals[u_ptr+1] & GETNODE_MASK;
            if (v_hrnode & HVI_LEFT_BOUNDARY_MASK) {
                v_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = vals[v_ptr+1] & GETNODE_MASK;
                if (u_right < v_hrnode) {
                    u_ptr += 2;
                    continue;
                }
                else if (u_hrnode > v_right) {
                    v_ptr += 2;
                    continue;
                }
                else {
                    res += min(u_right, v_right) - max(u_hrnode, v_hrnode) + 1;
                    if (u_right < v_right) u_ptr += 2;
                    else if (u_right > v_right) v_ptr += 2;
                    else {
                        u_ptr += 2;
                        v_ptr += 2;
                    }
                }
            }
            else {
                if (v_hrnode < u_hrnode) v_ptr++;
                else if (v_hrnode > u_right) u_ptr++;
                else {
                    res++;
                    v_ptr++;
                }
            }
        }
        else {
            if (v_hrnode & HVI_LEFT_BOUNDARY_MASK) {
                v_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
                uint32_t v_right = vals[v_ptr+1] & GETNODE_MASK;
                if (u_hrnode < v_hrnode) u_ptr++;
                else if (u_hrnode > v_right) v_ptr += 2;
                else {
                    res++;
                    u_ptr++;
                }
            }
            else {
                if (u_hrnode < v_hrnode) u_ptr++;
                else if (v_hrnode < u_hrnode) v_ptr++;
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

size_t HVI::cnt_tri_merge() const {
    size_t res = 0;
    for (node u = 0; u < num_nodes; ++u) {
        node curr_head = 0;
        for (size_t pos = inds[u]; pos < inds[u+1]; pos++) {
            HybridVertexInterval hrnode = vals[pos];
            if (hrnode & HVI_LEFT_BOUNDARY_MASK) {
                curr_head = hrnode ^ HVI_LEFT_BOUNDARY_MASK;
                continue;
            }
            if (hrnode & HVI_RIGHT_BOUNDARY_MASK) {
                node next = curr_head;
                node right = hrnode & GETNODE_MASK;
                while (next <= right) {
                    res += cnt_common_neighbor(u, next);
                    next++;
                }    
            }
            else {
                res += cnt_common_neighbor(u, hrnode);
            }
        }
    }
    return res;
}

void HVI::page_rank(vector<double> &res, double epsilon, size_t max_iter) {
    res = vector<double>(num_nodes, (double) 1);
	vector<double> tmp_res(num_nodes, 0);
    vector<size_t> degs(num_nodes, 0);
	// double diff = (double) num_nodes;
	size_t iter = 0;
    for (size_t u = 0; u < num_nodes; u++) {
        for (size_t i = inds[u]; i < inds[u+1];) {
            if (vals[i] & HVI_LEFT_BOUNDARY_MASK) {
                degs[u] += ((vals[i+1] & GETNODE_MASK) - (vals[i] & GETNODE_MASK) + 1);
                i += 2;
            }
            else {
                degs[u]++;
                i++;
            }
        }
    }
	// while (diff > epsilon && iter < max_iter) {
    while (iter < max_iter) {
		iter++;
		// diff = 0;
		swap(res, tmp_res);
        for (size_t u = 0; u < num_nodes; u++) {
			tmp_res[u] /= degs[u];
		}
		for (size_t u = 0; u < num_nodes; u++) {
			res[u] = 0;
            if (inds[u] == inds[u+1]) continue;
			for (size_t j = inds[u]; j < inds[u+1];) {
                if (vals[j] & HVI_LEFT_BOUNDARY_MASK) {
                    size_t left = vals[j] & GETNODE_MASK, right = vals[j+1] & GETNODE_MASK;
                    res[u] += accumulate(tmp_res.begin() + left, tmp_res.begin() + right + 1, (double) 0);
                    j += 2;
                }
                else {
                    res[u] += tmp_res[vals[j]];
                    j++;
                }
			}
			// diff += fabs(res[u] - tmp_res[u] * degs[u]);
		}
	}
    printf("-------- PageRank value --------\n");
	for (size_t u = 0; u < 5; u++) {
		printf("Node %lu, pr value: %.3lf.\n", u, res[u]);
	}
} 