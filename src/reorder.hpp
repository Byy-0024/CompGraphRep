#ifndef REORDER_HPP
#define REORDER_HPP
# include "Graph.hpp"
# include "Interval.hpp"
# include <omp.h>

typedef int weight;

void save_ordering_as_binfile(const std::vector<node> &origin_to_new, const std::string &binfilename);

class WeightedGraph {
public:
	WeightedGraph(std::vector<std::vector<std::pair<node, weight>>>& edges);
	~WeightedGraph();
	void get_ordering_greedy(std::vector<node> &origin_to_new);
	void maximun_cycle_cover_greedy(std::vector<std::pair<node, node>> &res);
	void maximum_perfect_matching_greedy(std::vector<std::pair<node, node>> &res);
	void expand_cycle_cover_to_ordering(std::vector<std::pair<node, node>> &CC, std::vector<node> &origin_to_new);
	void expand_perfect_matching_to_ordering(std::vector<std::pair<node, node>> &PM, std::vector<node> &origin_to_new);
    weight get_weight(node u, node v);

private:
	std::vector<std::vector<std::pair<node, weight>>> adj_list;
	size_t num_nodes, num_edges;
};

#endif // REORDER_HPP