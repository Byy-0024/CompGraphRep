#ifndef GRAPH_HPP
#define GRAPH_HPP
#pragma once
#include "util.hpp"


// #define ROULETTE_WHEEL_SAMPLING
#define COMPUTE_EXACT_INTERSECTION
using namespace std;
using namespace chrono;

static string dict_path = "./data/";
static vector<pair<string, string>> files = {
	{"arabic", "web-arabic-2005.mtx"},
	{"berkstan", "web-BerkStan-dir.edges"},
	{"bnu", "bn-human-BNU_1_0025864_session_1-bg.edges"},
	{"google", "web-google-dir.edges"},
	{"indochina", "web-indochina-2004-all.mtx"},
	{"ip", "tech-ip.edges"},
	{"it", "web-it-2004.mtx"},
	{"it-all", "web-it-2004-all.mtx"},
	{"jung", "bn-human-Jung2015_M87113878.edges"},
	{"ldoor", "sc-ldoor.mtx"},
	{"msdoor", "sc-msdoor.mtx"},
	{"nasa", "sc-nasasrb.mtx"},
	{"stanford", "web-Stanford.mtx"},
	{"uk", "web-uk-2005.mtx"},
	{"uk-all", "web-uk-2002-all.mtx"}
};

static string ordering_root = "./ordering/";
static set<string> ordering_methods = {"Greedy", "MWPM", "Serdyukov"};

class Graph {
public:
	Graph() {};
	Graph(string path);
	Graph(vector<vector<node>> &_adj_list);

	node get_one_neighbor(node u, int k) const;
	size_t get_num_nodes() const;
	size_t get_num_edges() const;
	size_t get_number_of_neighbors(node u) const;
	size_t size_in_bytes() const;
	
	bool has_undirected_edge(node u, node v) const;
	bool has_directed_edge(node u, node v) const;

	ull cnt_common_neighbors(node u, node v) const;
	ull cnt_common_neighbors(node u, const vector<node> &nodes) const;
	ull conditional_cnt_common_neighbor(node u, node v, node mu);
	ull cnt_tri_merge();
	void create_dag();
	void get_neighbors(node u, vector<node>& neis) const;
	void get_unseen_neighbors(node u, vector<bool> &seen, vector<node> &neis) const;
	void get_two_hop_neis(node u, vector<node> &res) const;
	void get_common_neighbors(node u, node v, vector<node>& res) const;
	void get_common_neighbors(node u, const vector<node> &nodes, vector<node> &res) const;
	void print_neis(node u);
	void reorder(vector<node> &labels);
	void save(string path);
	void bfs(node u, vector<int> &bfs_rank);
	void page_rank(vector<double> &res, double epsilon, size_t max_iter);

private:
	vector<vector<node>> adj_list;
	size_t num_nodes, num_edges, num_iso_nodes;
};

class WeightedGraph {
public:
	WeightedGraph(vector<vector<pair<node, size_t>>>& edges);
	~WeightedGraph();
	size_t get_weight(node u, node v);
	void get_ordering_greedy(vector<node> &origin_to_new);
	void maximun_cycle_cover_greedy(vector<pair<node, node>> &res);
	void maximum_perfect_matching_greedy(vector<pair<node, node>> &res);
	void expand_cycle_cover_to_ordering(vector<pair<node, node>> &CC, vector<node> &origin_to_new);
	void expand_perfect_matching_to_ordering(vector<pair<node, node>> &PM, vector<node> &origin_to_new);

private:
	vector<vector<pair<node, weight>>> adj_list;
	size_t num_nodes, num_edges;
};

#endif // GRAPH_HPP