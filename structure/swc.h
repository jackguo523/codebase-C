#ifndef STIM_SWC_H
#define STIM_SWC_H

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

//STIM includes
#include <stim/math/vec3.h>
#include <stim/parser/parser.h>

namespace stim {
	namespace swc_tree {
		template <typename T>
		class swc_node {

		protected:
			enum neuronal_type { SWC_UNDEFINED, SWC_SOMA, SWC_AXON, SWC_DENDRITE, SWC_APICAL_DENDRITE, SWC_FORK_POINT, SWC_END_POINT, SWC_CUSTOM };		// eight types
			enum node_information { INTEGER_LABEL, NEURONAL_TYPE, X_COORDINATE, Y_COORDINATE, Z_COORDINATE, RADIUS, PARENT_LABEL }; 

		public:
			int idx;							// the index of current node start from 1(original index from the file)
			neuronal_type type;					// the type of neuronal segmemt
			stim::vec3<T> point;				// the point coordinates 
			T radius;							// the radius at current node
			int parent_idx;						// parent idx of current node, -1 when it is origin(soma)
			int level;							// tree level
			std::vector<int> son_idx;			// son idx of current node

			swc_node() {						// default constructor
				idx = -1;						// set to default -1
				parent_idx = -1;				// set to origin -1
				level = -1;						// set to default -1
				type = SWC_UNDEFINED;			// set to undefined type
				radius = 0;						// set to 0
			}
			
			void get_node(std::string line) {	// get information from the node point we got
				
				// create a vector to store the information of one node point
				std::vector<std::string> p = stim::parser::split(line, ' ');

				// for each information contained in the node point we got
				for (unsigned int i = 0; i < p.size(); i++) {
					std::stringstream ss(p[i]);	// create a stringstream object for casting

					// store different information
					switch (i) {
					case INTEGER_LABEL:
						ss >> idx;				// cast the stringstream to the correct numerical value
						break;
					case NEURONAL_TYPE:
						int tmp_type;
						ss >> tmp_type;			// cast the stringstream to the correct numerical value
						type = (neuronal_type)tmp_type;
						break;
					case X_COORDINATE:
						T tmp_X;
						ss >> tmp_X;			// cast the stringstream to the correct numerical value
						point[0] = tmp_X;		// store the X_coordinate in vec3[0]
						break;
					case Y_COORDINATE:
						T tmp_Y;
						ss >> tmp_Y;			// cast the stringstream to the correct numerical value
						point[1] = tmp_Y;		// store the Y_coordinate in vec3[1]
						break;
					case Z_COORDINATE:
						T tmp_Z;
						ss >> tmp_Z;			// cast the stringstream to the correct numerical value
						point[2] = tmp_Z;		// store the Z_coordinate in vec3[2]
						break;
					case RADIUS:
						ss >> radius;			// cast the stringstream to the correct numerical value
						break;
					case PARENT_LABEL:
						ss >> parent_idx;		// cast the stringstream to the correct numerical value
						break;
					}
				}
			}
		};
	}		// end of namespace swc_tree

	template <typename T>
	class swc {
	public:
		std::vector< typename swc_tree::swc_node<T> > node;
		std::vector< std::vector<int> > E;			// list of nodes

		swc() {};									// default constructor

		// load the swc as tree nodes
		void load(std::string filename) {

			// load the file
			std::ifstream infile(filename.c_str());

			// if the file is invalid, throw an error
			if (!infile) {
				std::cerr << "STIM::SWC Error loading file" << filename << std::endl;
				exit(-1);
			}

			std::string line;
			// skip comment
			while (getline(infile, line)) {
				if ('#' == line[0] || line.empty())			// if it is comment line or empty line
					continue;								// keep read
				else
					break;
			}

			unsigned int l = 0;				// number of nodes

			// get rid of the first/origin node
			swc_tree::swc_node<T> new_node;
			new_node.get_node(line);
			l++;
			node.push_back(new_node);		// push back the first node

			getline(infile, line);			// get a new line
			// keep reading the following node point information as string
			while (!line.empty()) {			// test for the last empty line
				l++;						// still remaining node to be read

				swc_tree::swc_node<T> next_node;
				next_node.get_node(line);
				node.push_back(next_node);

				getline(infile, line);		// get a new line
			}
		}

		// read the head comment from swc file
		void read_comment(std::string filename) {

			// load the file
			std::ifstream infile(filename.c_str());

			// if the file is invalid, throw an error
			if (!infile) {
				std::cerr << "STIM::SWC Error loading file" << filename << std::endl;
				exit(1);
			}

			std::string line;
			while (getline(infile, line)) {
				if ('#' == line[0] || line.empty()) {
					std::cout << line << std::endl;			// print the comment line by line
				}
				else
					break;									// break when reaches to node information
			}
		}

		// link those nodes to create a tree
		void create_tree() {
			unsigned n = node.size();								// get the total number of node point
			int cur_level = 0;
			
			// build the origin(soma)
			node[0].level = cur_level;

			// go through follow nodes
			for (unsigned i = 1; i < n; i++) {
				if (node[i].parent_idx != node[i - 1].parent_idx)
					cur_level = node[node[i].parent_idx - 1].level + 1;
				node[i].level = cur_level;
				int tmp_parent_idx = node[i].parent_idx - 1;		// get the parent node loop idx of current node
				node[tmp_parent_idx].son_idx.push_back(i + 1);		// son_idx stores the real idx = loop idx + 1
			}
		}

		// create a new edge
		void create_up(std::vector<int>& edge,swc_tree::swc_node<T> cur_node, int target) {
			while (cur_node.parent_idx != target) {
				edge.push_back(cur_node.idx);
				cur_node = node[cur_node.parent_idx - 1];			// move to next node
			}
			edge.push_back(cur_node.idx);							// push back the start/end vertex of current edge
											
			std::reverse(edge.begin(), edge.end());					// follow the original flow direction
		}
		void create_down(std::vector<int>& edge, swc_tree::swc_node<T> cur_node, int j) {
			while (cur_node.son_idx.size() != 0) {
				edge.push_back(cur_node.idx);
				if (cur_node.son_idx.size() > 1)
					cur_node = node[cur_node.son_idx[j] - 1];		// move to next node
				else
					cur_node = node[cur_node.son_idx[0] - 1];
			}
			edge.push_back(cur_node.idx);							// push back the start/end vertex of current edge
		}

		// resample the tree-like SWC
		void resample() {
			unsigned n = node.size();

			std::vector<int> joint_node;
			for (unsigned i = 1; i < n; i++) {			// search all nodes(except the first one) to find joint nodes
				if (node[i].son_idx.size() > 1) {
					joint_node.push_back(node[i].idx);	// store the original index
				}
			}

			std::vector<int>  new_edge;					// new edge in the network
			
			n = joint_node.size();
			
			for (unsigned i = 0; i < n; i++) {			// for every joint nodes
				std::vector<int>  new_edge;									// new edge in the network
				swc_tree::swc_node<T> cur_node = node[joint_node[i] - 1];	// store current node
				
				// go up
				swc_tree::swc_node<T> tmp_node = node[cur_node.parent_idx - 1];	
				while (tmp_node.parent_idx != -1 && tmp_node.son_idx.size() == 1) {
					tmp_node = node[tmp_node.parent_idx - 1];
				}
				int target = tmp_node.parent_idx;							// store the go-up target
				create_up(new_edge, cur_node, target);
				E.push_back(new_edge);
				new_edge.clear();

				// go down
				unsigned t = cur_node.son_idx.size();
				for (unsigned j = 0; j < t; j++) {							// for every son node
					tmp_node = node[cur_node.son_idx[j] - 1];				// move down
					while (tmp_node.son_idx.size() == 1) {
						tmp_node = node[tmp_node.son_idx[0] - 1];
					}
					if (tmp_node.son_idx.size() == 0) {
						create_down(new_edge, cur_node, j);
						E.push_back(new_edge);
						new_edge.clear();
					}
				}
			}
		}

		// get points in one edge
		void get_points(int e, std::vector< stim::vec3<T> >& V) {
			V.resize(E[e].size());
			for (unsigned i = 0; i < E[e].size(); i++) {
				unsigned id = E[e][i] - 1;

				V[i][0] = node[id].point[0];
				V[i][1] = node[id].point[1];
				V[i][2] = node[id].point[2];
			}
		}

		// get radius information in one edge
		void get_radius(int e, std::vector<T>& radius) {
			radius.resize(E[e].size());
			for (unsigned i = 0; i < E[e].size(); i++) {
				unsigned id = E[e][i] - 1;

				radius[i] = node[id].radius;
			}
		}

		// return the number of point in swc
		unsigned int numP() {
			return node.size();
		}

		// return the number of edges in swc after resample
		unsigned int numE() {
			return E.size();
		}


	};
}		// end of namespace stim

#endif