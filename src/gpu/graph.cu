#include "graph.cuh"

using namespace std;

void cuda_csr_graph::init(const Graph& G) {
	const int n = G.get_num_nodes();
	const int m = G.get_num_edges();
	this->num_edges = m;
	this->num_nodes = n;
	cudaMalloc((void **)&this->d_adjacencyList, m * sizeof(node));
	cudaMalloc((void **)&this->d_edgesOffset, n * sizeof(size_t));
	cudaMalloc((void **)&this->d_edgesSize, n * sizeof(size_t));

	// Create CSR graph on device.
	vector<node> adjacencyList;
	vector<size_t> edgesOffset;
	vector<size_t> edgesSize;
	vector<node> neis;
	adjacencyList.reserve(G.get_num_edges());
	edgesOffset.reserve(G.get_num_nodes());
	edgesSize.reserve(G.get_num_nodes());
	for (node u = 0; u < G.get_num_nodes(); u++) {
		edgesOffset.push_back(adjacencyList.size());
		G.get_neis(u, neis);
		adjacencyList.insert(adjacencyList.end(), neis.begin(), neis.end());
		edgesSize.push_back(neis.size());
	}

	cudaMemcpy(this->d_adjacencyList, &adjacencyList[0], m * sizeof(node), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_edgesOffset, &edgesOffset[0], n * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_edgesSize, &edgesSize[0], n * sizeof(size_t), cudaMemcpyHostToDevice);
}

void cuda_csr_graph::free() {
	cudaFree(this->d_adjacencyList);
	cudaFree(this->d_edgesOffset);
	cudaFree(this->d_edgesSize);
}

void cuda_coo_graph::init(const Graph& G) {
	const int n = G.get_num_nodes();
	const int m = G.get_num_edges();
	this->num_edges = m;
	this->num_nodes = n;
	cudaMalloc((void **)&this->d_sourceNodesList, m * sizeof(node));
	cudaMalloc((void **)&this->d_targetNodesList, m * sizeof(node));
	cudaMalloc((void **)&this->d_edgesOffset, n * sizeof(size_t));
	cudaMalloc((void **)&this->d_edgesSize, n * sizeof(size_t));
	printf("Size of COO cuda graph:%.3f MB.\n", float(sizeof(node)*this->num_edges*2 + sizeof(size_t)*this->num_nodes*2) / (1<<20));

	// Create CSR graph on device.
	vector<node> sourceNodesList;
	vector<node> targetNodesList;
	vector<size_t> edgesOffset;
	vector<size_t> edgesSize;
	vector<node> neis;
	sourceNodesList.reserve(m);
	targetNodesList.reserve(m);
	edgesOffset.reserve(n);
	edgesSize.reserve(n);
	for (node u = 0; u < G.get_num_nodes(); u++) {
		edgesOffset.push_back(sourceNodesList.size());
		G.get_neis(u, neis);
		edgesSize.push_back(neis.size());
		if (neis.empty()) continue;
		targetNodesList.insert(targetNodesList.end(), neis.begin(), neis.end());
		sourceNodesList.resize(targetNodesList.size(), u);
	}
	ull res = 0;
	for (size_t i = 0; i < 5; i++) {
		ull cnt = 0;
		node u = sourceNodesList[i], v = targetNodesList[i];
		size_t u_ind = edgesOffset[u], v_ind = edgesOffset[v];
		while(u_ind < edgesOffset[u] + edgesSize[u] && v_ind < edgesOffset[v] + edgesSize[v]) {
			if (targetNodesList[u_ind] == targetNodesList[v_ind]) {
				cnt++;
				u_ind++;
				v_ind++;
			}
			else if (targetNodesList[u_ind] < targetNodesList[v_ind]) u_ind++;
			else v_ind++;
		}	
		res += cnt;
		printf("u: %u, v: %u, common neis: %llu.\n", u, v, cnt);
	}
	printf("Number of triangles: %llu.\n", res);

	cudaMemcpy(this->d_sourceNodesList, sourceNodesList.data(), sourceNodesList.size() * sizeof(node), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_targetNodesList, targetNodesList.data(), targetNodesList.size() * sizeof(node), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_edgesOffset, &edgesOffset[0], n * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_edgesSize, &edgesSize[0], n * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {  
        printf("cuda graph initialize failed with error code %s!\n", cudaGetErrorString(err)); 
    }
}

void cuda_coo_graph::free() {
	cudaFree(this->d_sourceNodesList);
	cudaFree(this->d_targetNodesList);
	cudaFree(this->d_edgesOffset);
	cudaFree(this->d_edgesSize);
}

void cuda_hvi_graph::init(const HVI& G) {
	const int n = G.get_num_nodes();
	const int m = G.get_num_hvi();
	this->num_hvi = m;
	this->num_nodes = n;
	cudaMalloc((void **)&this->d_HVIList, m * sizeof(HybridVertexInterval));
	cudaMalloc((void **)&this->d_HVIOffset, (n+1) * sizeof(size_t));

	// Create CSR graph on device.
	cudaMemcpy(this->d_HVIList, G.get_vals(), m * sizeof(HybridVertexInterval), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_HVIOffset, G.get_inds(), (n+1) * sizeof(size_t), cudaMemcpyHostToDevice);
}

void cuda_hvi_graph::free() {
	cudaFree(this->d_HVIList);
	cudaFree(this->d_HVIOffset);
}

void cuda_hvi_coo_graph::init(const HVI& G) {
	const int n = G.get_num_nodes();
	const int m = G.get_num_hvi();
	this->num_hvi = m;
	this->num_nodes = n;
	cudaMalloc((void **)&this->d_sourceNodesList, m * sizeof(node));
	cudaMalloc((void **)&this->d_HVIList, m * sizeof(HybridVertexInterval));
	cudaMalloc((void **)&this->d_HVIOffset, (n+1) * sizeof(size_t));

	// Create CSR graph on device.
	cudaMemcpy(this->d_HVIList, G.get_vals(), m * sizeof(HybridVertexInterval), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_HVIOffset, G.get_inds(), (n+1) * sizeof(size_t), cudaMemcpyHostToDevice);
	vector<node> sourceNodesList;
	sourceNodesList.reserve(m);
	for (node u = 0; u < n; u++) {
		for (size_t i = G.get_inds()[u]; i < G.get_inds()[u+1]; ++i){
			sourceNodesList.emplace_back(u);
		}
	}
	cudaMemcpy(this->d_sourceNodesList, &sourceNodesList[0], m * sizeof(node), cudaMemcpyHostToDevice);
}

void cuda_hvi_coo_graph::free() {
	cudaFree(this->d_HVIList);
	cudaFree(this->d_HVIOffset);
	cudaFree(this->d_sourceNodesList);
}

__device__ void cnt_intersection (node u, node v, node *targetNodesList, size_t *edgesOffset, size_t *edgesSize, ull *res) {
	ull cnt = 0;
	size_t u_ind = edgesOffset[u], v_ind = edgesOffset[v];
	while(u_ind < edgesOffset[u] + edgesSize[u] && v_ind < edgesOffset[v] + edgesSize[v]) {
		if (targetNodesList[u_ind] == targetNodesList[v_ind]) {
			cnt++;
			u_ind++;
			v_ind++;
		}
		else if (targetNodesList[u_ind] < targetNodesList[v_ind]) u_ind++;
		else v_ind++;
	}
	*res += cnt;
}

__device__ void cnt_intersection_hvi (node u, node v, HybridVertexInterval *HVIList, size_t *HVIOffset, ull *res) {
	ull cnt = 0;
	size_t u_ptr = HVIOffset[u], v_ptr = HVIOffset[v];
	while (u_ptr < HVIOffset[u+1] && v_ptr < HVIOffset[v+1]) {
		HybridVertexInterval u_hrnode = HVIList[u_ptr], v_hrnode = HVIList[v_ptr];
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
			uint32_t u_right = HVIList[u_ptr+1] & GETNODE_MASK;
			if (v_hrnode & HVI_LEFT_BOUNDARY_MASK) {
				v_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
				uint32_t v_right = HVIList[v_ptr+1] & GETNODE_MASK;
				if (u_right < v_hrnode) {
					u_ptr += 2;
					continue;
				}
				else if (u_hrnode > v_right) {
					v_ptr += 2;
					continue;
				}
				else {
					cnt += min(u_right, v_right) - max(u_hrnode, v_hrnode) + 1;
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
					cnt++;
					v_ptr++;
				}
			}
		}
		else {
			if (v_hrnode & HVI_LEFT_BOUNDARY_MASK) {
				v_hrnode ^= HVI_LEFT_BOUNDARY_MASK;
				uint32_t v_right = HVIList[v_ptr+1] & GETNODE_MASK;
				if (u_hrnode < v_hrnode) u_ptr++;
				else if (u_hrnode > v_right) v_ptr += 2;
				else {
					cnt++;
					u_ptr++;
				}
			}
			else {
				if (u_hrnode < v_hrnode) u_ptr++;
				else if (v_hrnode < u_hrnode) v_ptr++;
				else {
					cnt++;
					u_ptr++;
					v_ptr++;
				}
			}
		}
	}
	*res += cnt;
}

__device__ void has_edge(node u, node v, node *adjacencyList, size_t *edgesOffset, size_t *edgesSize, bool *res) {
	*res = false;
	if (edgesSize[u] < edgesSize[v]) {
			node *__first = &adjacencyList[edgesOffset[u]], *__last = &adjacencyList[edgesOffset[u]+edgesSize[u]];
			int __len = __last - __first;
			while (__len > 0) {
				int __half = __len >> 1;
				node *__middle = __first + __half;
				if (*__middle < v){
					__first = __middle;
					++__first;
					__len = __len - __half - 1;
				}
				else __len = __half;
			}
			if (*__first == v && __first != __last) *res = true;
		}
		else {
			node *__first = &adjacencyList[edgesOffset[v]], *__last = &adjacencyList[edgesOffset[v]+edgesSize[v]];
			int __len = __last - __first;
			while (__len > 0) {
				int __half = __len >> 1;
				node *__middle = __first + __half;
				if (*__middle < u){
					__first = __middle;
					++__first;
					__len = __len - __half - 1;
				}
				else __len = __half;
			}
			if (*__first == u && __first != __last) *res = true;
		}
}

__device__ void has_edge (node u, node v, HybridVertexInterval *HVIList, size_t *HVIOffset, bool *res) {
	*res = false;
	if (HVIOffset[u] == HVIOffset[u+1] || HVIOffset[v] == HVIOffset[v+1]) return;
	if (HVIOffset[u+1] - HVIOffset[u] < HVIOffset[v+1] - HVIOffset[v]) {
		HybridVertexInterval *__first = &HVIList[HVIOffset[u]], *__last = &HVIList[HVIOffset[u+1]];
		int __len = __last - __first;
		while (__len > 0) {
			int __half = __len >> 1;
			HybridVertexInterval *__middle = __first + __half; 
			if ((*__middle & GETNODE_MASK) < v) {
				__first = __middle;
				++__first;
				__len = __len - __half - 1;
			}
			else __len = __half;
		}
		size_t lower_bound_pos = __first - HVIList;
		if (lower_bound_pos == HVIOffset[u+1]) return;
		if (HVIList[lower_bound_pos] & HVI_RIGHT_BOUNDARY_MASK) {
			*res = true;
		}
		else {
			if((HVIList[lower_bound_pos] & GETNODE_MASK) == v) *res = true;
		}
	}
	else {
		HybridVertexInterval *__first = &HVIList[HVIOffset[v]], *__last = &HVIList[HVIOffset[v+1]];
		int __len = __last - __first;
		while (__len > 0) {
			int __half = __len >> 1;
			HybridVertexInterval *__middle = __first + __half; 
			if ((*__middle & GETNODE_MASK) < u) {
				__first = __middle;
				++__first;
				__len = __len - __half - 1;
			}
			else __len = __half;
		}
		size_t lower_bound_pos = __first - HVIList;
		if (lower_bound_pos == HVIOffset[v+1]) return;
		if (HVIList[lower_bound_pos] & HVI_RIGHT_BOUNDARY_MASK) {
			*res = true;
		}
		else {
			if((HVIList[lower_bound_pos] & GETNODE_MASK) == u) *res = true;
		}
	}
}