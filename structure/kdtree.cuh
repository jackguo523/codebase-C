// right now the size of CUDA STACK is set to 50, increase it if you mean to make deeper tree
// data should be stored in row-major
// x1,x2,x3,x4,x5......
// y1,y2,y3,y4,y5......
// ....................
// ....................

#ifndef KDTREE_H
#define KDTREE_H
#define stack_size 50

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <vector>
#include <cstring>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <stim/cuda/cudatools/error.h>
#include <stim/visualization/aabbn.h>

namespace stim {
	namespace cpu_kdtree {
		template<typename T, int D>											// typename refers to float or double while D refers to dimension of points
		struct point {
			T dim[D];														// create a structure to store every one input point
		};

		template<typename T>
		class cpu_kdnode {
		public:
			cpu_kdnode() {														// constructor for initializing a kdnode
				parent = NULL;												// set every node's parent, left and right kdnode pointers to NULL
				left = NULL;
				right = NULL;
				parent_idx = -1;											// set parent node index to default -1
				left_idx = -1;
				right_idx = -1;
				split_value = -1;											// set split_value to default -1
			}
			int idx;														// index of current node
			int parent_idx, left_idx, right_idx;							// index of parent, left and right nodes
			cpu_kdnode *parent, *left, *right;									// parent, left and right kdnodes
			T split_value;													// splitting value of current node
			std::vector <size_t> indices;									// it indicates the points' indices that current node has 
			size_t level;													// tree level of current node
		};
	}				// end of namespace cpu_kdtree

	template <typename T>
	struct cuda_kdnode {
		int parent, left, right;														
		T split_value;
		size_t num_index;																					// number of indices it has
		int index;																							// the beginning index
		size_t level;
	};

	template <typename T, int D>
    __device__ T gpu_distance(cpu_kdtree::point<T, D> &a, cpu_kdtree::point<T, D> &b) {
		T distance = 0;

		for (size_t i = 0; i < D; i++) {
			T d = a.dim[i] - b.dim[i];
			distance += d*d;
		}
		return distance;
	}
	
	template <typename T, int D>
	__device__ void search_at_node(cuda_kdnode<T> *nodes, size_t *indices, cpu_kdtree::point<T, D> *d_reference_points, int cur, cpu_kdtree::point<T, D> &d_query_point, size_t *d_index, T *d_distance, int *d_node) {
		T best_distance = FLT_MAX;
		size_t best_index = 0;

		while (true) {																						// break until reach the bottom
			int split_axis = nodes[cur].level % D;
			if (nodes[cur].left == -1) {																	// check whether it has left node or not
				*d_node = cur;
				for (int i = 0; i < nodes[cur].num_index; i++) {
					size_t idx = indices[nodes[cur].index + i];
					T dist = gpu_distance<T, D>(d_query_point, d_reference_points[idx]);
					if (dist < best_distance) {
						best_distance = dist;
						best_index = idx;
					}
				}
			break;
			}
			else if (d_query_point.dim[split_axis] < nodes[cur].split_value) {								// jump into specific son node
				cur = nodes[cur].left;
			}
			else {
				cur = nodes[cur].right;
			}
		}
		*d_distance = best_distance;
		*d_index = best_index;
	}
	
	template <typename T, int D>
	__device__ void search_at_node_range(cuda_kdnode<T> *nodes, size_t *indices, cpu_kdtree::point<T, D> *d_reference_points, cpu_kdtree::point<T, D> &d_query_point, int cur, T range, size_t *d_index, T *d_distance, size_t id, int *next_nodes, int *next_search_nodes, int *Judge) {
		T best_distance = FLT_MAX;
		size_t best_index = 0;

		int next_nodes_pos = 0;																				// initialize pop out order index
		next_nodes[id * stack_size + next_nodes_pos] = cur;															// find data that belongs to the very specific thread
		next_nodes_pos++;

		while (next_nodes_pos) {
			int next_search_nodes_pos = 0;																	// record push back order index
			while (next_nodes_pos) {
				cur = next_nodes[id * stack_size + next_nodes_pos - 1];												// pop out the last push in one and keep poping out
				next_nodes_pos--;
				int split_axis = nodes[cur].level % D;

				if (nodes[cur].left == -1) {
					for (int i = 0; i < nodes[cur].num_index; i++) {
						int idx = indices[nodes[cur].index + i];											// all indices are stored in one array, pick up from every node's beginning index
						T d = gpu_distance<T>(d_query_point, d_reference_points[idx]);
						if (d < best_distance) {
							best_distance = d;
							best_index = idx;
						}
					}
				}
				else {
					T d = d_query_point.dim[split_axis] - nodes[cur].split_value;

					if (fabs(d) > range) {
						if (d < 0) {
							next_search_nodes[id * stack_size + next_search_nodes_pos] = nodes[cur].left;
							next_search_nodes_pos++;
						}
						else {
							next_search_nodes[id * stack_size + next_search_nodes_pos] = nodes[cur].right;
							next_search_nodes_pos++;
						}
					}
					else {
						next_search_nodes[id * stack_size + next_search_nodes_pos] = nodes[cur].right;
						next_search_nodes_pos++;
						next_search_nodes[id * stack_size + next_search_nodes_pos] = nodes[cur].left;
						next_search_nodes_pos++;
						if (next_search_nodes_pos > stack_size) {
							printf("Thread conflict might be caused by thread %d, so please try smaller input max_tree_levels\n", id);
							(*Judge)++;
						}
					}
				}
			}
			for (int i = 0; i < next_search_nodes_pos; i++)
				next_nodes[id * stack_size + i] = next_search_nodes[id * stack_size + i];
			next_nodes_pos = next_search_nodes_pos;										
		}
		*d_distance = best_distance;
		*d_index = best_index;
	}
	
	template <typename T, int D>
	__device__ void search(cuda_kdnode<T> *nodes, size_t *indices, cpu_kdtree::point<T, D> *d_reference_points, cpu_kdtree::point<T, D> &d_query_point, size_t *d_index, T *d_distance, size_t id, int *next_nodes, int *next_search_nodes, int *Judge) {
		int best_node = 0;
		T best_distance = FLT_MAX;
		size_t best_index = 0;
		T radius = 0;

		search_at_node<T, D>(nodes, indices, d_reference_points, 0, d_query_point, &best_index, &best_distance, &best_node);
		radius = sqrt(best_distance);																															// get range
		int cur = best_node;

		while (nodes[cur].parent != -1) {
			int parent = nodes[cur].parent;
			int split_axis = nodes[parent].level % D;

			T tmp_dist = FLT_MAX;
			size_t tmp_idx;
			if (fabs(nodes[parent].split_value - d_query_point.dim[split_axis]) <= radius) {
				if (nodes[parent].left != cur)
					search_at_node_range(nodes, indices, d_reference_points, d_query_point, nodes[parent].left, radius, &tmp_idx, &tmp_dist, id, next_nodes, next_search_nodes, Judge);
				else
					search_at_node_range(nodes, indices, d_reference_points, d_query_point, nodes[parent].right, radius, &tmp_idx, &tmp_dist, id, next_nodes, next_search_nodes, Judge);
			}
			if (tmp_dist < best_distance) {
				best_distance = tmp_dist;
				best_index = tmp_idx;
			}
			cur = parent;
		}
		*d_distance = sqrt(best_distance);
		*d_index = best_index;
	}
	
	template <typename T, int D>
	__global__ void search_batch(cuda_kdnode<T> *nodes, size_t *indices, cpu_kdtree::point<T, D> *d_reference_points, cpu_kdtree::point<T, D> *d_query_points, size_t d_query_count, size_t *d_indices, T *d_distances, int *next_nodes, int *next_search_nodes, int *Judge) {
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= d_query_count) return;																														 // avoid segfault

		search<T, D>(nodes, indices, d_reference_points, d_query_points[idx], &d_indices[idx], &d_distances[idx], idx, next_nodes, next_search_nodes, Judge);    // every query points are independent
	}
	
	template <typename T, int D>
	void search_stream(cuda_kdnode<T> *d_nodes, size_t *d_index, cpu_kdtree::point<T, D> *d_reference_points, cpu_kdtree::point<T, D> *query_stream_points, size_t stream_count, size_t *indices, T *distances) {
		unsigned int threads = (unsigned int)(stream_count > 1024 ? 1024 : stream_count);
		unsigned int blocks = (unsigned int)(stream_count / threads + (stream_count % threads ? 1 : 0));

		cpu_kdtree::point<T, D> *d_query_points;	
		size_t *d_indices;
		T *d_distances;

		int *next_nodes;																																	
		int *next_search_nodes;
		
		HANDLE_ERROR(cudaMalloc((void**)&d_query_points, sizeof(T) * stream_count * D));
		HANDLE_ERROR(cudaMalloc((void**)&d_indices, sizeof(size_t) * stream_count));
		HANDLE_ERROR(cudaMalloc((void**)&d_distances, sizeof(T) * stream_count));
		HANDLE_ERROR(cudaMalloc((void**)&next_nodes, threads * blocks * stack_size * sizeof(int)));																	
		HANDLE_ERROR(cudaMalloc((void**)&next_search_nodes, threads * blocks * stack_size * sizeof(int)));	
		HANDLE_ERROR(cudaMemcpy(d_query_points, query_stream_points, sizeof(T) * stream_count * D, cudaMemcpyHostToDevice));

		int *Judge = NULL;

		search_batch<<<blocks, threads>>> (d_nodes, d_index, d_reference_points, d_query_points, stream_count, d_indices, d_distances, next_nodes, next_search_nodes, Judge);

		if(Judge == NULL) {
			HANDLE_ERROR(cudaMemcpy(indices, d_indices, sizeof(size_t) * stream_count, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(distances, d_distances, sizeof(T) * stream_count, cudaMemcpyDeviceToHost));
		}

		HANDLE_ERROR(cudaFree(next_nodes));
		HANDLE_ERROR(cudaFree(next_search_nodes));
		HANDLE_ERROR(cudaFree(d_query_points));
		HANDLE_ERROR(cudaFree(d_indices));
		HANDLE_ERROR(cudaFree(d_distances));
	}

	template <typename T, int D = 3>										// set dimension of data to default 3
	class kdtree {
	protected:
		int current_axis;													// current judging axis
		int n_id;															// store the total number of nodes
		std::vector < typename cpu_kdtree::point<T, D> > *tmp_points;			// transfer or temperary points
		std::vector < typename cpu_kdtree::point<T, D> > cpu_tmp_points;		// for cpu searching
		cpu_kdtree::cpu_kdnode<T> *root;											// root node
		static kdtree<T, D> *cur_tree_ptr;
		#ifdef __CUDACC__
			cuda_kdnode<T> *d_nodes;                                                    																		 
			size_t *d_index;
			cpu_kdtree::point<T, D>* d_reference_points;
			size_t npts;
			int num_nodes;
		#endif
	public:
		kdtree() {														// constructor for creating a cpu_kdtree
			cur_tree_ptr = this;											// create  a class pointer points to the current class value
			n_id = 0;														// set total number of points to default 0
		}
		
		~kdtree() {											  			// destructor of cpu_kdtree
			std::vector <cpu_kdtree::cpu_kdnode<T>*> next_nodes;
			next_nodes.push_back(root);
			while (next_nodes.size()) {
				std::vector <cpu_kdtree::cpu_kdnode<T>*> next_search_nodes;
				while (next_nodes.size()) {
					cpu_kdtree::cpu_kdnode<T> *cur = next_nodes.back();
					next_nodes.pop_back();
					if (cur->left)
						next_search_nodes.push_back(cur->left);
					if (cur->right)
						next_search_nodes.push_back(cur->right);
					delete cur;
				}
				next_nodes = next_search_nodes;
			}
			root = NULL;
			#ifdef __CUDACC__
			HANDLE_ERROR(cudaFree(d_nodes));
			HANDLE_ERROR(cudaFree(d_index));
			HANDLE_ERROR(cudaFree(d_reference_points));
			#endif
		}
		
		void cpu_create(std::vector < typename cpu_kdtree::point<T, D> > &reference_points, size_t max_levels) {									
			tmp_points = &reference_points;
			root = new cpu_kdtree::cpu_kdnode<T>();									// initializing the root node
			root->idx = n_id++;												// the index of root is 0
			root->level = 0;												// tree level begins at 0
			root->indices.resize(reference_points.size());					// get the number of points
			for (size_t i = 0; i < reference_points.size(); i++) {
				root->indices[i] = i;										// set indices of input points
			}
			std::vector <cpu_kdtree::cpu_kdnode<T>*> next_nodes;					// next nodes
			next_nodes.push_back(root);										// push back the root node
			while (next_nodes.size()) {
				std::vector <cpu_kdtree::cpu_kdnode<T>*> next_search_nodes;			// next search nodes
				while (next_nodes.size()) {									// two same WHILE is because we need to make a new vector to store nodes for search
					cpu_kdtree::cpu_kdnode<T> *current_node = next_nodes.back();	// handle node one by one (right first) 
					next_nodes.pop_back();									// pop out current node in order to store next round of nodes
					if (current_node->level < max_levels) {					
						if (current_node->indices.size() > 1) {				// split if the nonleaf node contains more than one point
							cpu_kdtree::cpu_kdnode<T> *left = new cpu_kdtree::cpu_kdnode<T>();
							cpu_kdtree::cpu_kdnode<T> *right = new cpu_kdtree::cpu_kdnode<T>();
							left->idx = n_id++;								// set the index of current node's left node
							right->idx = n_id++;							
							split(current_node, left, right);				// split left and right and determine a node
							std::vector <size_t> temp;						// empty vecters of int
							//temp.resize(current_node->indices.size());
							current_node->indices.swap(temp);				// clean up current node's indices
							current_node->left = left;
							current_node->right = right;
							current_node->left_idx = left->idx;				
							current_node->right_idx = right->idx;					
							if (right->indices.size())
								next_search_nodes.push_back(right);			// left pop out first
							if (left->indices.size())
								next_search_nodes.push_back(left);	
						}
					}
				}
				next_nodes = next_search_nodes;								// go deeper within the tree
			}
		}
		
		static bool sort_points(const size_t a, const size_t b) {									// create functor for std::sort
			std::vector < typename cpu_kdtree::point<T, D> > &pts = *cur_tree_ptr->tmp_points;			// put cur_tree_ptr to current input points' pointer
			return pts[a].dim[cur_tree_ptr->current_axis] < pts[b].dim[cur_tree_ptr->current_axis];
		}
		
		void split(cpu_kdtree::cpu_kdnode<T> *cur, cpu_kdtree::cpu_kdnode<T> *left, cpu_kdtree::cpu_kdnode<T> *right) {
			std::vector < typename cpu_kdtree::point<T, D> > &pts = *tmp_points;
			current_axis = cur->level % D;												// indicate the judicative dimension or axis
			std::sort(cur->indices.begin(), cur->indices.end(), sort_points);			// using SortPoints as comparison function to sort the data
			size_t mid_value = cur->indices[cur->indices.size() / 2];                   // odd in the mid_value, even take the floor
			cur->split_value = pts[mid_value].dim[current_axis];						// get the parent node
			left->parent = cur;                                                         // set the parent of the next search nodes to current node
			right->parent = cur;
			left->level = cur->level + 1;												// level + 1
			right->level = cur->level + 1;
			left->parent_idx = cur->idx;                                                // set its parent node's index
			right->parent_idx = cur->idx;                                            
			for (size_t i = 0; i < cur->indices.size(); i++) {							// split into left and right half-space one by one
				size_t idx = cur->indices[i];
				if (pts[idx].dim[current_axis] < cur->split_value)
					left->indices.push_back(idx);
				else
					right->indices.push_back(idx);
			}
		}
		
		/// create a KD-tree given a pointer to an array of reference points and the number of reference points
		/// @param h_reference_points is a host array containing the reference points in (x0, y0, z0, ...., ) order
		/// @param reference_count is the number of reference point in the array	 
		/// @param max_levels is the deepest number of tree levels allowed
		void create(T *h_reference_points, size_t reference_count, size_t max_levels) {
			#ifdef __CUDACC__
			if (max_levels > 10) {
				std::cout<<"The max_tree_levels should be smaller!"<<std::endl;
				exit(1);
			}		
			//bb.init(&h_reference_points[0]);
			//aaboundingboxing<T, D>(bb, h_reference_points, reference_count);

			std::vector < typename cpu_kdtree::point<T, D>> reference_points(reference_count);																				// restore the reference points in particular way
			for (size_t j = 0; j < reference_count; j++)
				for (size_t i = 0; i < D; i++)
					reference_points[j].dim[i] = h_reference_points[j * D + i];																																// creating a tree on cpu
			(*this).cpu_create(reference_points, max_levels);																											// building a tree on cpu
			cpu_kdtree::cpu_kdnode<T> *d_root = (*this).get_root();
			num_nodes = (*this).get_num_nodes();
			npts = reference_count;																												// also equals to reference_count

			HANDLE_ERROR(cudaMalloc((void**)&d_nodes, sizeof(cuda_kdnode<T>) * num_nodes));																		// copy data from host to device
			HANDLE_ERROR(cudaMalloc((void**)&d_index, sizeof(size_t) * npts));
			HANDLE_ERROR(cudaMalloc((void**)&d_reference_points, sizeof(cpu_kdtree::point<T, D>) * npts));

			std::vector < cuda_kdnode<T> > tmp_nodes(num_nodes);																									
			std::vector <size_t> indices(npts);
			std::vector <cpu_kdtree::cpu_kdnode<T>*> next_nodes;
			size_t cur_pos = 0;
			next_nodes.push_back(d_root);
			while (next_nodes.size()) {
				std::vector <typename cpu_kdtree::cpu_kdnode<T>*> next_search_nodes;
				while (next_nodes.size()) {
					cpu_kdtree::cpu_kdnode<T> *cur = next_nodes.back();
					next_nodes.pop_back();
					int id = cur->idx;																															// the nodes at same level are independent
					tmp_nodes[id].level = cur->level;
					tmp_nodes[id].parent = cur->parent_idx;
					tmp_nodes[id].left = cur->left_idx;
					tmp_nodes[id].right = cur->right_idx;
					tmp_nodes[id].split_value = cur->split_value;
					tmp_nodes[id].num_index = cur->indices.size();																								// number of index
					if (cur->indices.size()) {
						for (size_t i = 0; i < cur->indices.size(); i++)
							indices[cur_pos + i] = cur->indices[i];

						tmp_nodes[id].index = (int)cur_pos;																										// beginning index of reference_points that every bottom node has
						cur_pos += cur->indices.size();																											// store indices continuously for every query_point
					}
					else {
						tmp_nodes[id].index = -1;
					}

					if (cur->left)
						next_search_nodes.push_back(cur->left);

					if (cur->right)
						next_search_nodes.push_back(cur->right);
				}
				next_nodes = next_search_nodes;
			}
			HANDLE_ERROR(cudaMemcpy(d_nodes, &tmp_nodes[0], sizeof(cuda_kdnode<T>) * tmp_nodes.size(), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_index, &indices[0], sizeof(size_t) * indices.size(), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_reference_points, &reference_points[0], sizeof(cpu_kdtree::point<T, D>) * reference_count, cudaMemcpyHostToDevice));

			#else
			std::vector < typename cpu_kdtree::point<T, D> > reference_points(reference_count);		// restore the reference points in particular way
			for (size_t j = 0; j < reference_count; j++)
				for (size_t i = 0; i < D; i++)
					reference_points[j].dim[i] = h_reference_points[j * D + i];
			cpu_create(reference_points, max_levels);
			cpu_tmp_points = *tmp_points;

			#endif
		}
		
		int get_num_nodes() const {														// get the total number of nodes
			return n_id; 
		}
		
		cpu_kdtree::cpu_kdnode<T>* get_root() const {											// get the root node of tree
			return root; 
		}
        
		T cpu_distance(const cpu_kdtree::point<T, D> &a, const cpu_kdtree::point<T, D> &b) {
			T distance = 0;

			for (size_t i = 0; i < D; i++) {
				T d = a.dim[i] - b.dim[i];
				distance += d*d;
			}
			return distance;
		}
		
		void cpu_search_at_node(cpu_kdtree::cpu_kdnode<T> *cur, const cpu_kdtree::point<T, D> &query, size_t *index, T *distance, cpu_kdtree::cpu_kdnode<T> **node) {
			T best_distance = FLT_MAX;                                              // initialize the best distance to max of floating point
			size_t best_index = 0;
			std::vector < typename cpu_kdtree::point<T, D> > pts = cpu_tmp_points;
			while (true) {
				size_t split_axis = cur->level % D;
				if (cur->left == NULL) {                                            // risky but acceptable, same goes for right because left and right are in same pace
					*node = cur;													// pointer points to a pointer
					for (size_t i = 0; i < cur->indices.size(); i++) {
						size_t idx = cur->indices[i];
						T d = cpu_distance(query, pts[idx]);						// compute distances
						/// if we want to compute k nearest neighbor, we can input the last resul
						/// (last_best_dist < dist < best_dist) to select the next point until reaching to k
						if (d < best_distance) {
							best_distance = d;
							best_index = idx;                                       // record the nearest neighbor index
						}
					}
					break;                                                          // find the target point then break the loop
				}
				else if (query.dim[split_axis] < cur->split_value) {				// if it has son node, visit the next node on either left side or right side
					cur = cur->left;
				}
				else {
					cur = cur->right;
				}
			}
			*index = best_index;
			*distance = best_distance;
		} 
		
		void cpu_search_at_node_range(cpu_kdtree::cpu_kdnode<T> *cur, const cpu_kdtree::point<T, D> &query, T range, size_t *index, T *distance) {
			T best_distance = FLT_MAX;                                              // initialize the best distance to max of floating point
			size_t best_index = 0;
			std::vector < typename cpu_kdtree::point<T, D> > pts = cpu_tmp_points;
			std::vector < typename cpu_kdtree::cpu_kdnode<T>*> next_node;
			next_node.push_back(cur);
			while (next_node.size()) {
				std::vector<typename cpu_kdtree::cpu_kdnode<T>*> next_search;
				while (next_node.size()) {
					cur = next_node.back();                                         
					next_node.pop_back();
					size_t split_axis = cur->level % D;
					if (cur->left == NULL) {
						for (size_t i = 0; i < cur->indices.size(); i++) {
							size_t idx = cur->indices[i];
							T d = cpu_distance(query, pts[idx]);
							if (d < best_distance) {
								best_distance = d;
								best_index = idx;
							}
						}
					}
					else {
						T d = query.dim[split_axis] - cur->split_value;				// computer distance along specific axis or dimension
						/// there are three possibilities: on either left or right, and on both left and right
						if (fabs(d) > range) {										// absolute value of floating point to see if distance will be larger that best_dist
							if (d < 0)
								next_search.push_back(cur->left);                   // every left[split_axis] is less and equal to cur->split_value, so it is possible to find the nearest point in this region
							else
								next_search.push_back(cur->right);
						}
						else {                                                      // it is possible that nereast neighbor will appear on both left and right 
							next_search.push_back(cur->left);
							next_search.push_back(cur->right);
						}
					}
				}
				next_node = next_search;                                            // pop out at least one time                                  
			}
			*index = best_index;
			*distance = best_distance;
		}
		
		void cpu_search(T *h_query_points, size_t query_count, size_t *h_indices, T *h_distances) {
			/// first convert the input query point into specific type
			cpu_kdtree::point<T, D> query;
			for (size_t j = 0; j < query_count; j++) {
				for (size_t i = 0; i < D; i++)
					query.dim[i] = h_query_points[j * D + i];
				/// find the nearest node, this will be the upper bound for the next time searching
				cpu_kdtree::cpu_kdnode<T> *best_node = NULL;
				T best_distance = FLT_MAX;
				size_t best_index = 0;
				T radius = 0;																				// radius for range                                                                           
				cpu_search_at_node(root, query, &best_index, &best_distance, &best_node);                   // simple search to rougly determine a result for next search step
				radius = sqrt(best_distance);                                                               // It is possible that nearest will appear in another region
				/// find other possibilities
				cpu_kdtree::cpu_kdnode<T> *cur = best_node;
				while (cur->parent != NULL) {																// every node that you pass will be possible to be the best node
					/// go up
					cpu_kdtree::cpu_kdnode<T> *parent = cur->parent;                                                // travel back to every node that we pass through
					size_t split_axis = (parent->level) % D;
					/// search other nodes
					size_t tmp_index;
					T tmp_distance = FLT_MAX;
					if (fabs(parent->split_value - query.dim[split_axis]) <= radius) {
						/// search opposite node
						if (parent->left != cur)
							cpu_search_at_node_range(parent->left, query, radius, &tmp_index, &tmp_distance);        // to see whether it is its mother node's left son node
						else
							cpu_search_at_node_range(parent->right, query, radius, &tmp_index, &tmp_distance);
					}
					if (tmp_distance < best_distance) {
						best_distance = tmp_distance;
						best_index = tmp_index;
					}
					cur = parent;
				}
				h_indices[j] = best_index;
				h_distances[j] = best_distance;
			}
		}

		/// search the KD tree for nearest neighbors to a set of specified query points
		/// @param h_query_points an array of query points in (x0, y0, z0, ...) order 
		/// @param query_count is the number of query points 
		/// @param indices are the indices to the nearest reference point for each query points 
		/// @param distances is an array containing the distance between each query point and the nearest reference point
		void search(T *h_query_points, size_t query_count, size_t *indices, T *distances) {
			#ifdef __CUDACC__
			std::vector < typename cpu_kdtree::point<T, D> > query_points(query_count);
			for (size_t j = 0; j < query_count; j++)
				for (size_t i = 0; i < D; i++)
					query_points[j].dim[i] = h_query_points[j * D + i];

			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			
			size_t query_memory = D * sizeof(T) * query_count;
			size_t N = 3 * query_memory / prop.totalGlobalMem;					//consider index and distance, roughly 3 times
			if (N > 1) {
				N++;
				size_t stream_count = query_count / N;
				for (size_t n = 0; n < N; n++) {
					size_t query_stream_start = n * stream_count;
					search_stream(d_nodes, d_index, d_reference_points, &query_points[query_stream_start], stream_count, &indices[query_stream_start], &distances[query_stream_start]);
				}
				size_t stream_remain_count = query_count - N * stream_count;
				if (stream_remain_count > 0) {
					size_t query_remain_start = N * stream_count;
					search_stream(d_nodes, d_index, d_reference_points, &query_points[query_remain_start], stream_remain_count, &indices[query_remain_start], &distances[query_remain_start]);
				}
			}
			else {
				unsigned int threads = (unsigned int)(query_count > 1024 ? 1024 : query_count);
				unsigned int blocks = (unsigned int)(query_count / threads + (query_count % threads ? 1 : 0));

				cpu_kdtree::point<T, D> *d_query_points;																												// create a pointer pointing to query points on gpu
				size_t *d_indices;
				T *d_distances;

				int *next_nodes;																																	// create two STACK-like array
				int *next_search_nodes;

				int *Judge = NULL;																																	// judge variable to see whether one thread is overwrite another thread's memory																						
		
				HANDLE_ERROR(cudaMalloc((void**)&d_query_points, sizeof(T) * query_count * D));
				HANDLE_ERROR(cudaMalloc((void**)&d_indices, sizeof(size_t) * query_count));
				HANDLE_ERROR(cudaMalloc((void**)&d_distances, sizeof(T) * query_count));
				HANDLE_ERROR(cudaMalloc((void**)&next_nodes, threads * blocks * stack_size * sizeof(int)));																	// STACK size right now is 50, you can change it if you mean to
				HANDLE_ERROR(cudaMalloc((void**)&next_search_nodes, threads * blocks * stack_size * sizeof(int)));	
				HANDLE_ERROR(cudaMemcpy(d_query_points, &query_points[0], sizeof(T) * query_count * D, cudaMemcpyHostToDevice));

				search_batch<<<blocks, threads>>> (d_nodes, d_index, d_reference_points, d_query_points, query_count, d_indices, d_distances, next_nodes, next_search_nodes, Judge);

				if (Judge == NULL) {																																// do the following work if the thread works safely
					HANDLE_ERROR(cudaMemcpy(indices, d_indices, sizeof(size_t) * query_count, cudaMemcpyDeviceToHost));
					HANDLE_ERROR(cudaMemcpy(distances, d_distances, sizeof(T) * query_count, cudaMemcpyDeviceToHost));
				}

				HANDLE_ERROR(cudaFree(next_nodes));
				HANDLE_ERROR(cudaFree(next_search_nodes));
				HANDLE_ERROR(cudaFree(d_query_points));
				HANDLE_ERROR(cudaFree(d_indices));
				HANDLE_ERROR(cudaFree(d_distances));
			}

			#else
			cpu_search(h_query_points, query_count, indices, distances);

			#endif

		}

	};				//end class kdtree

	template <typename T, int D>
	kdtree<T, D>* kdtree<T, D>::cur_tree_ptr = NULL;												// definition of cur_tree_ptr pointer points to the current class

}				//end namespace stim
#endif