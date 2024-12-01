#ifndef COMPRESSGRAPH_HPP
#define COMPRESSGRAPH_HPP
#pragma once
#include "Graph.hpp"

 class CompressGraph{
    public:
        CompressGraph(const std::string csr_vlist_binfile, const std::string csr_elist_binfile, const size_t _num_nodes);

        // Operators.
        const int* get_csr_vlist() const;
        const node* get_csr_elist() const;
        bool has_directed_edge(node u, node v) const;
        bool has_edge(node u, node v) const;
        bool cmp_node_with_rule(node u, node r) const;
        node get_rule_min(node r) const;
        node get_rule_max(node r) const;
        size_t cnt_common_neis(node u, node v) const;
        size_t cnt_common_neis(const std::vector<node> &neis1, const std::vector<node> &neis2) const;
        size_t get_deg(node u) const;
        size_t get_num_edges() const;
        size_t get_num_nodes() const;
        size_t get_num_rules() const;
        size_t size_in_bytes() const;
        void decode_rule(node r, std::vector<node> &neis) const;
        void get_common_neis(const std::vector<node> &neis1, const std::vector<node> &neis2, std::vector<node> &res) const;
        void get_neis(node u, std::vector<node> &neis) const;
        void print_neis(node u) const;

        // Algorithms.
        size_t tc();
        ull mc(node u);
        ull subgraph_matching(const Graph &q);
        void BKP(std::vector<node> &P, std::vector<node> &X, ull &res);
        void bfs(node u, std::vector<size_t> &bfs_rank); 
        void cc(std::vector<node> &component_id);
        void core_decomposition(std::vector<size_t> &core_numbers);
        void dfs_helper(node u, std::vector<bool> &visited, std::vector<int> &new2origin) const;
        void get_dfs_order(node u, std::vector<int> &new2origin) const;
        void page_rank(std::vector<double> &res, double epsilon, size_t max_iter);
        void subgraph_matching_helper(const Graph &q, std::unordered_set<node> &partial_matching, std::vector<int> &join_order, std::vector<std::vector<node>> &inter_res, ull &res);

    private:
        size_t num_nodes, num_rules, num_edges;
        std::vector<int> csr_vlist;
        std::vector<node> csr_elist;
        friend struct CompressGrpahIterator;
 };

 struct CompressGraphIterator{
    CompressGraphIterator(const CompressGraph &g, node u);
    CompressGraphIterator& operator++();
    void decode_rule(node r);
    node get_val();

    const int *csr_vlist;
    const node *csr_elist;
    size_t curr_ptr;
    size_t end_ptr;
    size_t num_nodes;
    size_t buffer_ptr;
    std::vector<node> buffer;
};
#endif