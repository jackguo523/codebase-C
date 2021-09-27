#ifndef JACK_NETWORK_H
#define JACK_NETWORK_H

#include "centerline.h"

#include <stim/visualization/obj.h>
#include <stim/visualization/swc.h>
#include <stim/math/circle.h>
#include <stim/structures/kdtree.cuh>


// *****help function*****

template<typename T>
CUDA_CALLABLE T gaussian(T x, T std = 25.0f) {
	return exp(-x / (2.0f * std * std));
}

#ifdef __CUDACC__
template<typename T>
__global__ void find_metric(T* M, size_t n, T* D, T sigma) {
	size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;		// segfault
	M[x] = 1.0f - gaussian<T>(D[x], sigma);
}
#endif

namespace stim {

	template<typename T>
	class network {
	public:
		// define edge class that extends centerline class with radius information
		class edge : public stim::centerline<T> {
		protected:
			std::vector<T> R;	// radius at each point on current edge

		public:
			unsigned int v[2];		// unique idx for starting and ending point
			using stim::centerline<T>::d;
			using stim::centerline<T>::C;


			/// constructors
			// empty constructor
			edge() : stim::centerline<T>() {
				v[0] = UINT_MAX;		// set to default value, risky!
				v[1] = UINT_MAX;
			}

			// constructor that contructs an edge based on a centerline
			edge(stim::centerline<T> c) : stim::centerline<T>(c) {
				size_t num = c.size();
				R.resize(num);
			}

			// constructor that constructs an edge based on a centerline and a list of radii
			edge(stim::centerline<T> c, std::vector<T> r) : stim::centerline<T>(c) {
				R = r;			// copy radii
			}

			// constructor that constructs an edge based on a list of points and a list of radii
			edge(std::vector<stim::vec3<T> > c, std::vector<T> r) : stim::centerline<T>(c) {
				R = r;			// copy radii
			}


			/// basic operations
			// get radius
			T r(size_t idx) {
				return R[idx];
			}

			// set radius
			void set_r(T value) {
				size_t num = R.size();
				for (size_t i = 0; i < num; i++)
					R[i] = value;
			}
			void set_r(size_t idx, T value) {
				R[idx] = value;
			}
			void set_r(std::vector<T> value) {
				size_t num = value.size();
				for (size_t i = 0; i < num; i++)
					R[i] = value[i];
			}

			// push back a new radius
			void push_back_r(T value) {
				R.push_back(value);
			}


			/// vector operation
			// insert a new point and radius at specific location
			void insert(size_t p, stim::vec3<T> v, T r) {
				centerline<T>::insert(p, v);		// insert a new point on current centerline

				R.insert(R.begin() + p, r);			// insert a new radius for that point
			}

			// amend a existing point and radius at specific location
			void amend(size_t p, stim::vec3<T> v, T r) {
				centerline<T>::amend(p, v);

				R[p] = r;
			}

			// reverse the order of an edge
			edge reverse() {
				centerline<T> new_centerline = (*this).centerline<T>::reverse();
				std::vector<T> new_radius = R;
				std::reverse(new_radius.begin(), new_radius.end());

				edge result(new_centerline, new_radius);

				return result;
			}

			// copy edge to array
			void edge_to_array(T* a) {
				edge b = (*this);
				size_t n = b.size();
				for (size_t i = 0; i < n; i++) {
					a[i * 3 + 0] = b[i][0];
					a[i * 3 + 1] = b[i][1];
					a[i * 3 + 2] = b[i][2];
				}
			}


			/// arithmetic operations
			// '+' operation
			edge operator+(edge e) const {
				edge result(*this);
				size_t num = e.size();
				for (size_t i = 0; i < num; i++) {
					result.push_back(e[i]);
					result.push_back_r(e.R[i]);
				}

				return result;
			}


			/// advanced operations
			// concatenate two edges from specific point, The deference between function "+" and "concatenate" is that this requires that new edges should share a joint point
			edge concatenate(edge e2, size_t p1, size_t p2) {
				edge e1 = *this;
				size_t num1 = e1.size();		// get the number of points
				size_t num2 = e2.size();
				centerline<T> new_centerline;
				std::vector<T> new_radius;

				// four situations
				if (p1 == 0) {
					if (p2 == 0)
						e2 = e2.reverse();
					for (size_t i = 0; i < num2 - 1; i++) {
						new_centerline.push_back(e2[i]);
						new_radius.push_back(e2.R[i]);
					}
					for (size_t i = 0; i < num1; i++) {
						new_centerline.push_back(e1[i]);
						new_radius.push_back(e1.R[i]);
					}
				}
				else {
					if (p2 != 0)
						e2 = e2.reverse();
					for (size_t i = 0; i < num1 - 1; i++) {
						new_centerline.push_back(e1[i]);
						new_radius.push_back(e1.R[i]);
					}
					for (size_t i = 0; i < num2; i++) {
						new_centerline.push_back(e2[i]);
						new_radius.push_back(e2.R[i]);
					}
				}

				edge result(new_centerline, new_radius);

				return result;
			}

			// split current edge at specific position
			std::vector<edge> split(size_t idx) {

				// can't update v!!!
				std::vector<centerline<T> > tmp;
				tmp = (*this).centerline<T>::split(idx);			// split current edge in terms of centerline
				size_t num = tmp.size();
				std::vector<edge> result(num);						// construct a list of edges
				for (size_t i = 0; i < num; i++) {
					edge new_edge(tmp[i]);							// construct new edge based on one centerline
					result[i] = new_edge;
				}

				for (size_t i = 0; i < num; i++) {					// for every edge
					for (size_t j = 0; j < result[i].size(); j++) {	// for every point on that edge
						result[i].R[j] = R[j + i * idx];			// update radius information
					}
				}

				return result;
			}

			// resample current edge
			edge resample(T spacing) {
				edge result(centerline<T>::resample(spacing));		// resample current edge and output as a new edge
				result.v[0] = v[0];									// updates unique indices
				result.v[1] = v[1];

				return result;
			}

			// compute a circle that represent the shape of cylinder cross section at point idx, (TGC -> truncated generalized cones)
			stim::circle<T> circ(size_t idx) {

				stim::circle<T> c;			// create a circle to orient for finding the circle plane at point idx
				c.rotate(d(idx));			// rotate the circle
				stim::vec3<T> U = c.U;		// copy the frenet frame vector

				return stim::circle<T>(C[idx], R[idx], d(idx), U);
			}

			/// output operation
			// output the edge information as a string
			std::string str() {
				std::stringstream ss;
				ss << "(" << centerline<T>::size() << ")\tl = " << this->length() << "\t" << v[0] << "----" << v[1];
				return ss.str();
			}

			/// operator for writing the edge information into a binary .nwt file.
			friend std::ofstream& operator<<(std::ofstream& out, edge& e) {
				out.write(reinterpret_cast<const char*>(&e.v[0]), sizeof(unsigned int));	// write the starting point.
				out.write(reinterpret_cast<const char*>(&e.v[1]), sizeof(unsigned int));	// write the ending point.
				size_t sz = e.size();				// write the number of point in the edge.
				out.write(reinterpret_cast<const char*>(&sz), sizeof(unsigned int));
				for (size_t i = 0; i < sz; i++) {	// write each point
					stim::vec3<T> point = e[i];
					out.write(reinterpret_cast<const char*>(&point[0]), 3 * sizeof(T));
					out.write(reinterpret_cast<const char*>(&e.R[i]), sizeof(T));			// write the radius
				}
				return out;	// return stream
			}

			/// operator for reading an edge from a binary .nwt file.
			friend std::ifstream& operator >> (std::ifstream& in, edge& e) {
				unsigned int v0, v1, sz;
				in.read(reinterpret_cast<char*>(&v0), sizeof(unsigned int));	// read the staring point.
				in.read(reinterpret_cast<char*>(&v1), sizeof(unsigned int));	// read the ending point
				in.read(reinterpret_cast<char*>(&sz), sizeof(unsigned int));	// read the number of points in the edge

				std::vector<stim::vec3<T> > p(sz);
				std::vector<T> r(sz);
				for (size_t i = 0; i < sz; i++) {	// set the points and radii to the newly read values
					stim::vec3<T> point;
					in.read(reinterpret_cast<char*>(&point[0]), 3 * sizeof(T));
					p[i] = point;
					T mag;
					in.read(reinterpret_cast<char*>(&mag), sizeof(T));
					r[i] = mag;
				}
				e = edge(p, r);
				e.v[0] = v0; e.v[1] = v1;
				return in;
			}
		};

		// define vertex class that extends vec3 class with connectivity information
		class vertex : public stim::vec3<T> {
		public:
			std::vector<unsigned int> e[2];	// incoming and outgoing edges of that vertex
			using stim::vec3<T>::ptr;
			unsigned D = 0;					// vertex degree

			/// constructors
			// empty constructor
			vertex() : stim::vec3<T>() {
			}

			// constructor that constructs a vertex based on a vec3 vector
			vertex(stim::vec3<T> v) : stim::vec3<T>(v) {
			}

			stim::vec3<T> getPosition() {
				return stim::vec3<T>(ptr[0], ptr[1], ptr[2]);
			}

			/// output operation
			// output the vertex information as a string
			std::string str() {
				std::stringstream ss;
				ss << "\t(x, y, z) = " << stim::vec3<T>::str();

				if (e[0].size() > 0) {
					ss << "\t> ";
					for (size_t i = 0; i < e[0].size(); i++)
						ss << e[0][i] << " ";
				}
				if (e[1].size() > 0) {
					ss << "\t< ";
					for (size_t i = 0; i < e[1].size(); i++)
						ss << e[1][i] << " ";
				}

				return ss.str();
			}

			/// operator for writing the vector into the stream;
			friend std::ofstream& operator<<(std::ofstream& out, const vertex& v) {
				unsigned int s0, s1;
				s0 = v.e[0].size();
				s1 = v.e[1].size();
				out.write(reinterpret_cast<const char*>(&v.ptr[0]), 3 * sizeof(T));		// write physical vertex location
				out.write(reinterpret_cast<const char*>(&s0), sizeof(unsigned int));			// write the number of "outgoing edges"
				out.write(reinterpret_cast<const char*>(&s1), sizeof(unsigned int));			// write the number of "incoming edges"	
				if (s0 != 0)
					out.write(reinterpret_cast<const char*>(&v.e[0][0]), sizeof(unsigned int)*v.e[0].size());	// write the "outgoing edges"
				if (s1 != 0)
					out.write(reinterpret_cast<const char*>(&v.e[1][0]), sizeof(unsigned int)*v.e[1].size());	// write the "incoming edges"
				return out;
			}

			/// operator for reading the vector out of the stream;
			friend std::ifstream& operator >> (std::ifstream& in, vertex& v) {
				in.read(reinterpret_cast<char*>(&v[0]), 3 * sizeof(T));				// read the physical position
				unsigned int s[2];
				in.read(reinterpret_cast<char*>(&s[0]), 2 * sizeof(unsigned int));	// read the sizes of incoming and outgoing edge arrays

				std::vector<unsigned int> one(s[0]);
				std::vector<unsigned int> two(s[1]);
				v.e[0] = one;
				v.e[1] = two;
				if (one.size() != 0)
					in.read(reinterpret_cast<char*>(&v.e[0][0]), s[0] * sizeof(unsigned int));		// read the arrays of "outgoing edges"
				if (two.size() != 0)
					in.read(reinterpret_cast<char*>(&v.e[1][0]), s[1] * sizeof(unsigned int));		// read the arrays of "incoming edges"

				v.D = one.size() + two.size();
				return in;
			}
		};

	protected:

		std::vector<edge> E;	// list of edges
		std::vector<vertex> V;	// list of vertices

	public:

		/// constructors
		// empty constructor
		network() {
		}

		// constructor with a file to load
		network(std::string fileLocation) {
			load_obj(fileLocation);
		}

		// constructor that constructs a network based on lists of vertices and edges
		network(std::vector<edge> nE, std::vector<vertex> nV) {
			E = nE;
			V = nV;
		}


		/// basic operations
		// get the number of edges
		size_t edges() {
			return E.size();
		}
		size_t edges(size_t f) {
			return E[f].size();
		}

		// get the number of vertices
		size_t vertices() {
			return V.size();
		}

		// get the radius at specific point
		T r(size_t f, size_t p) {		// edge f, point p
			return E[f].r(p);
		}
		T r(size_t c) {					// vertex c
			T result;
			if (V[c].e[0].size()) {				// if this vertex has outgoing edges
				size_t f = V[c].e[0][0];		// get the index of first outgoing edge of this vertex
				result = r(f, 0);				// this vertex should be the starting point of that edge
			}
			else {								// if this vertex only has incoming edges
				size_t f = V[c].e[1][0];		// get the index of first incoming edge of this vertex
				result = r(f, E[f].size() - 1);	// this vertex should be the ending point of that edge
			}

			return result;
		}

		// get the average radius of one specific edge
		T ar(size_t f) {
			T result = 0.0f;
			size_t num = E[f].size();
			for (size_t i = 0; i < num; i++)
				result += E[f].r(i);
			result = result / num;

			return result;
		}

		// get the length of edge "f"
		T length(size_t f) {
			return E[f].length();
		}

		// copy specific edge/point
		edge get_edge(size_t f) {
			return E[f];
		}
		stim::vec3<T> get_point(size_t f, size_t p) {
			return E[f][p];
		}

		// copy specific vertex
		vertex get_vertex(size_t c) {
			return V[c];
		}

		// get the first/last point index
		unsigned get_first_vertex(unsigned f) {
			return E[f].v[0];
		}
		unsigned get_last_vertex(unsigned f) {
			return E[f].v[1];
		}

		// get boundary/pendant vertices
		std::vector<size_t> pendant() {
			std::vector<size_t> result;

			for (size_t i = 0; i < V.size(); i++)
				if (V[i].D < 2)
					result.push_back(i);

			return result;
		}

		// find near points that connected to current point (f, p)
		std::vector<stim::vec3<T> > near_points(size_t f, size_t p) {
			
			std::vector<stim::vec3<T> > result;
			size_t idx;

			if (p == 0 || p == E[f].size() - 1) {		// if current point is the first/last point of fiber 'f'
				if (p == 0)		// get the vertex index
					idx = E[f].v[0];
				else
					idx = E[f].v[1];

				for (size_t i = 0; i < V[idx].e[0].size(); i++)		// for outgoing fibers
					result.push_back(E[V[idx].e[0][i]][1]);
				for (size_t i = 0; i < V[idx].e[1].size(); i++) {	// for incoming fibers
					size_t num = E[V[idx].e[1][i]].size();	// get the number of points on the fiber
					result.push_back(E[V[idx].e[1][i]][num - 2]);
				}
			}
			else {		// if current point is an interval point of fiber f
				result.push_back(E[f][p - 1]);
				result.push_back(E[f][p + 1]);
			}

			return result;
		}

		// set radius for specific point on edge "f"
		void set_r(size_t f, size_t p, T value) {
			E[f].set_r(p, value);
		}
		void set_r(size_t f, T value) {
			E[f].set_r(value);
		}
		void set_r(size_t f, std::vector<T> value) {
			E[f].set_r(value);
		}
		// set the radius of a vertex
		void set_vr(size_t v, T value) {
			size_t num1 = V[v].e[0].size();
			size_t num2 = V[v].e[1].size();
			size_t f;

			if (num1) {				// list 1
				for (size_t i = 0; i < num1; i++) {
					f = V[v].e[0][i];
					set_r(f, 0, value);
				}
			}
			if (num2) {				// list 2
				for (size_t i = 0; i < num2; i++) {
					f = V[v].e[1][i];
					set_r(f, E[f].size() - 1, value);
				}
			}
		}

		// set specific point
		void set_point(size_t f, size_t p, stim::vec3<T> newv) {
			
			size_t idx;

			if (p == 0 || p == E[f].size() - 1) {		// if current point is the first/last point of fiber 'f'
				if (p == 0)		// get the vertex index
					idx = E[f].v[0];
				else
					idx = E[f].v[1];
				for (size_t d = 0; d < 3; d++)		// set new position
					V[idx].ptr[d] = newv[d];
				for (size_t i = 0; i < V[idx].e[0].size(); i++)	// for outgoing fibers
					E[V[idx].e[0][i]].amend(0, newv, E[V[idx].e[0][i]].r(0));
				for (size_t i = 0; i < V[idx].e[1].size(); i++) {	// for incoming fibers
					size_t num = E[V[idx].e[1][i]].size();	// get the number of points on the fiber
					E[V[idx].e[1][i]].amend(num - 1, newv, E[V[idx].e[1][i]].r(num - 1));
				}
			}
			else {		// if current point is an interval point of fiber f
				E[f].amend(p, newv, E[f].r(p));
			}
		}

		// copy all points (coordinates) to 1D array
		void copy_to_array(T* dst) {
			size_t t = 0;								// indicator for points
			for (size_t i = 0; i < E.size(); i++) {
				for (size_t j = 0; j < E[i].size(); j++) {
					for (size_t k = 0; k < 3; k++) {
						dst[t * 3 + k] = E[i][j][k];
					}
					t++;			// next point
				}
			}
		}

		// get an average of branching index in the network
		T BranchingIndex() {
			T B = 0.0f;
			size_t num = V.size();
			for (size_t i = 0; i < num; i++)
				B += (T)(V[i].e[0].size() + V[i].e[1].size());

			B = B / (T)num;

			return B;
		}

		// get the number of branching points in the network
		size_t BranchP() {
			size_t B = 0;
			size_t c;
			size_t num = V.size();
			for (size_t i = 0; i < num; i++) {
				c = (V[i].e[0].size() + V[i].e[1].size());
				if (c > 2)
					B += 1;
			}

			return B;
		}

		// get the number of starting or ending points in the network
		size_t EndP() {
			size_t B = 0;
			size_t c;
			size_t num = V.size();
			for (size_t i = 0; i < num; i++) {
				c = (V[i].e[0].size() + V[i].e[1].size());
				if (c == 1)
					B += 1;
			}

			return B;
		}

		// get an average of fiber length in the network
		T Lengths() {
			std::vector<T> L;
			T sumLength = 0.0f;
			size_t num = E.size();
			for (size_t i = 0; i < num; i++) {
				L.push_back(E[i].length());
				sumLength += E(i).length();
			}
			T avg = sumLength / (T)num;

			return avg;
		}

		// get the total number of points in the network
		size_t total_points() {
			size_t n = 0;
			size_t num = E.size();
			for (size_t i = 0; i < num; i++)
				n += E[i].size();

			return n;
		}

		// get an average of tortuosities in the network
		T Tortuosities() {
			std::vector<T> t;
			std::vector<T> id1, id2;
			T distance;
			T tortuosity;
			T sumTortuosity = 0.0f;
			size_t num = E.size();
			for (size_t i = 0; i < num; i++) {
				id1 = E[i][0];						// get the starting point
				id2 = E[i][num - 1];				// get the ending point
				distance = (id1 - id2).len();
				if (distance > 0)
					tortuosity = E[i].length() / distance;	// tortuosity = edge length / edge displacement
				else
					tortuosity = 0.0f;
				t.push_back(tortuosity);
				sumTortuosity += tortuosity;
			}
			T avg = sumTortuosity / (T)num;

			return avg;
		}

		// get an average contraction of the network
		T Contraction() {
			std::vector<T> t;
			std::vector<T> id1, id2;							// starting and ending vertices of the edge
			T distance;
			T contraction;
			T sumContraction = 0.0f;
			size_t num = E.size();
			for (size_t i = 0; i < num; i++) {					// for each edge in the network
				id1 = E[i][0];									// get the edge starting point
				id2 = E[i][num - 1];							// get the edge ending point
				distance = (id1 - id2).len();                   // displacement between the starting and ending points
				contraction = distance / E[i].length();			// contraction = edge displacement / edge length
				t.push_back(contraction);
				sumContraction += contraction;
			}
			T avg = sumContraction / (T)num;

			return avg;
		}

		// get an average fractal dimension of the branches of the network
		T FractalDimensions() {
			std::vector<T> t;
			std::vector<T> id1, id2;							// starting and ending vertices of the edge
			T distance;
			T fract;
			T sumFractDim = 0.0f;
			size_t num = E.size();
			for (size_t i = 0; i < num; i++) {							// for each edge in the network
				id1 = E[i][0];											// get the edge starting point
				id2 = E[i][num - 1];									// get the edge ending point
				distance = (id1 - id2).len();							// displacement between the starting and ending points
				fract = std::log(distance) / std::log(E[i].length());	// fractal dimension = log(edge displacement) / log(edge length)
				t.push_back(sumFractDim);
				sumFractDim += fract;
			}
			T avg = sumFractDim / (T)num;

			return avg;
		}


		/// construct network from files
		// load network from OBJ files
		void load_obj(std::string filename) {
			stim::obj<T> O;				// create OBJ object
			O.load(filename);			// load OBJ file to an object

			size_t ii[2];				// starting/ending point index of one centerline/edge
			std::vector<size_t> index;	// added vertex index
			std::vector<size_t>::iterator it;	// iterator for searching
			size_t pos;					// position of added vertex

										// get the points
			for (size_t l = 1; l <= O.numL(); l++) {	// for every line of points
				std::vector<stim::vec<T> > tmp;			// temp centerline
				O.getLine(l, tmp);						// get points
				size_t n = tmp.size();
				std::vector<stim::vec3<T> > c(n);
				for (size_t i = 0; i < n; i++) {		// switch from vec to vec3
					for (size_t j = 0; j < 3; j++) {
						c[i][j] = tmp[i][j];
					}
				}

				centerline<T> C(c);				// construct centerline
				edge new_edge(C);				// construct edge without radii

				std::vector<unsigned> id;		// temp point index
				O.getLinei(l, id);				// get point index

				ii[0] = (size_t)id.front();
				ii[1] = (size_t)id.back();

				size_t num = new_edge.size();	// get the number of point on current edge

				// for starting point
				it = std::find(index.begin(), index.end(), ii[0]);
				if (it == index.end()) {		// new vertex
					vertex new_vertex = new_edge[0];
					bool flag = false;						// check repetition
					size_t j = 0;
					for (; j < V.size(); j++) {
						if (new_vertex == V[j]) {
							flag = true;
							break;
						}
					}
					if (!flag) {							// no repetition
						new_vertex.e[0].push_back(E.size());// push back the outgoing edge index
						new_vertex.D++;						// increase degree
						new_edge.v[0] = V.size();			// save the starting vertex index
						V.push_back(new_vertex);
						index.push_back(ii[0]);				// save the point index
					}
					else {
						V[j].e[0].push_back(E.size());
						V[j].D++;							// increase degree
						new_edge.v[0] = j;
					}
				}
				else {							// already added vertex
					pos = std::distance(index.begin(), it);	// find the added vertex position
					V[pos].e[0].push_back(E.size());		// push back the outgoing edge index
					V[pos].D++;					// increase degree
					new_edge.v[0] = pos;
				}

				// for ending point
				it = std::find(index.begin(), index.end(), ii[1]);
				if (it == index.end()) {		// new vertex
					vertex new_vertex = new_edge[num - 1];
					bool flag = false;						// check repetition
					size_t j = 0;
					for (; j < V.size(); j++) {
						if (new_vertex == V[j]) {
							flag = true;
							break;
						}
					}
					if (!flag) {							// no repetition
						new_vertex.e[1].push_back(E.size());// push back the incoming edge index
						new_vertex.D++;						// increase degree
						new_edge.v[1] = V.size();			// save the ending vertex index
						V.push_back(new_vertex);
						index.push_back(ii[1]);				// save the point index
					}
					else {
						V[j].e[1].push_back(E.size());
						V[j].D++;							// increase degree
						new_edge.v[1] = j;
					}
				}
				else {							// already added vertex
					pos = std::distance(index.begin(), it);	// find the added vertex position
					V[pos].e[1].push_back(E.size());		// push back the incoming edge index
					V[pos].D++;					// increase degree
					new_edge.v[1] = pos;
				}

				E.push_back(new_edge);
			}

			// get the radii
			if (O.numVT()) {		// copy radii information if provided
				std::vector<unsigned> id;		// a list stores the indices of points
				for (size_t i = 1; i <= O.numL(); i++) {
					id.clear();		// clear up temp for current round computation
					O.getLinei(i, id);
					size_t num = id.size();	// get the number of points
					T radius;
					for (size_t j = 0; j < num; j++) {
						radius = O.getVT(id[j] - 1)[0] / 2;		// hard-coded: radius = diameter / 2
						set_r(i - 1, j, radius);				// copy the radius
					}
				}
			}
		}

		// load network from SWC files (neuron)
		void load_swc(std::string filename) {
			stim::swc<T> S;				// create a SWC object
			S.load(filename);			// load data from SWC file to an object
			S.create_tree();			// link nodes according to connectivity as a tree
			S.resample();

			size_t i[2];				// starting/ending point index of one centerline/edge
			std::vector<size_t> index;	// added vertex index
			std::vector<size_t>::iterator it;	// iterator for searching
			size_t pos;					// position of added vertex

			for (size_t l = 0; l < S.numE(); l++) {
				std::vector<stim::vec3<T> > c;	// temp centerline
				S.get_points(l, c);

				centerline<T> C(c);				// construct centerline

				std::vector<T> radius;
				S.get_radius(l, radius);		// get radius

				edge new_edge(C, radius);		// construct edge
				size_t num = new_edge.size();	// get the number of point on current edge

				i[0] = S.E[l].front();
				i[1] = S.E[l].back();

				// for starting point
				it = std::find(index.begin(), index.end(), i[0]);
				if (it == index.end()) {		// new vertex
					vertex new_vertex = new_edge[0];
					new_vertex.e[0].push_back(E.size());	// push back the outgoing edge index
					new_edge.v[0] = V.size();				// save the starting vertex index
					V.push_back(new_vertex);
					index.push_back(i[0]);					// save the point index
				}
				else {							// already added vertex
					pos = std::distance(index.begin(), it);	// find the added vertex position
					V[pos].e[0].push_back(E.size());		// push back the outgoing edge index
					V[pos].D++;					// increase degree
					new_edge.v[0] = pos;
				}

				// for ending point
				it = std::find(index.begin(), index.end(), i[1]);
				if (it == index.end()) {		// new vertex
					vertex new_vertex = new_edge[num - 1];
					new_vertex.e[1].push_back(E.size());	// push back the incoming edge index
					new_edge.v[1] = V.size();				// save the ending vertex index
					V.push_back(new_vertex);
					index.push_back(i[1]);					// save the point index
				}
				else {							// already added vertex
					pos = std::distance(index.begin(), it);	// find the added vertex position
					V[pos].e[1].push_back(E.size());		// push back the incoming edge index
					V[pos].D++;					// increase degree
					new_edge.v[1] = pos;
				}

				E.push_back(new_edge);
			}
		}

		/*
		// load a network in text file to a network class
		void load_txt(std::string filename) {
		std::vector <std::string> file_contents;
		std::ifstream file(filename.c_str());
		std::string line;
		std::vector<size_t> id2vert;	// this list stores the vertex ID associated with each network vertex
		// for each line in the text file, store them as strings in file_contents
		while (std::getline(file, line)) {
		std::stringstream ss(line);
		file_contents.push_back(ss.str());
		}
		size_t numEdges = atoi(file_contents[0].c_str());	// number of edges in the network
		size_t I = atoi(file_contents[1].c_str());			// calculate the number of points3d on the first edge
		size_t count = 1; size_t k = 2;						// count is global counter through the file contents, k is for the vertices on the edges

		for (size_t i = 0; i < numEdges; i++) {
		// pre allocate a position vector p with number of points3d on the edge p
		std::vector<stim::vec<T> > p(0, I);
		// for each point on the nth edge
		for (size_t j = k; j < I + k; j++) {
		// split the points3d of floats with separator space and form a float3 position vector out of them
		p.push_back(std::split(file_contents[j], ' '));
		}
		count += p.size() + 1;	// increment count to point at the next edge in the network
		I = atoi(file_contents[count].c_str()); // read in the points3d at the next edge and convert it to an integer
		k = count + 1;
		edge new_edge = p;		// create an edge with a vector of points3d  on the edge
		E.push_back(new_edge);	// push the edge into the network
		}
		size_t numVertices = atoi(file_contents[count].c_str()); // this line in the text file gives the number of distinct vertices
		count = count + 1;			// this line of text file gives the first verrtex

		for (size_t i = 0; i < numVertices; i++) {
		vertex new_vertex = std::split(file_contents[count], ' ');
		V.push_back(new_vertex);
		count += atoi(file_contents[count + 1].c_str()) + 2; // Skip number of edge ids + 2 to point to the next vertex
		}
		}
		*/
		// load network from NWT files
		void load_nwt(std::string filename) {
			int dims[2];						// number of vertex, number of edges
			read_nwt_header(filename, &dims[0]);		// read header
			std::ifstream file;
			file.open(filename.c_str(), std::ios::in | std::ios::binary);		// skip header information.
			file.seekg(14 + 58 + 4 + 4, file.beg);
			vertex v;
			for (int i = 0; i < dims[0]; i++) {	// for every vertex, read vertex, add to network.
				file >> v;
				std::cerr << v.str() << std::endl;
				V.push_back(v);
			}

			std::cout << std::endl;
			for (int i = 0; i < dims[1]; i++) {	// for every edge, read edge, add to network.
				edge e;
				file >> e;
				std::cerr << e.str() << std::endl;
				E.push_back(e);
			}
			file.close();
		}

		// save network to NWT files
		void save_nwt(std::string filename) {
			write_nwt_header(filename);
			std::ofstream file;
			file.open(filename.c_str(), std::ios::out | std::ios::binary | std::ios::app);	///since we have written the header we are not appending.
			for (int i = 0; i < V.size(); i++) {	// look through the Vertices and write each one.
				file << V[i];
			}
			for (int i = 0; i < E.size(); i++) {	// loop through the Edges and write each one.
				file << E[i];
			}
			file.close();
		}

		/// NWT format functions
		void read_nwt_header(std::string filename, int *dims) {
			char magicString[14];		// id
			char desc[58];				// description
			int hNumVertices;			// #vert
			int hNumEdges;				// #edges
			std::ifstream file;			// create stream
			file.open(filename.c_str(), std::ios::in | std::ios::binary);
			file.read(reinterpret_cast<char*>(&magicString[0]), 14);		// read the file id.
			file.read(reinterpret_cast<char*>(&desc[0]), 58);				// read the description
			file.read(reinterpret_cast<char*>(&hNumVertices), sizeof(int));	// read the number of vertices
			file.read(reinterpret_cast<char*>(&hNumEdges), sizeof(int));	// read the number of edges
			file.close();								// close the file.
			dims[0] = hNumVertices;						// fill the returned reference.
			dims[1] = hNumEdges;
		}
		void write_nwt_header(std::string filename) {
			std::string magicString = "nwtFileFormat ";				// identifier for the file.
			std::string desc = "fileid(14B), desc(58B), #vertices(4B), #edges(4B): bindata";
			int hNumVertices = V.size();							// int byte header storing the number of vertices in the file
			int hNumEdges = E.size();								// int byte header storing the number of edges.
			std::ofstream file;
			file.open(filename.c_str(), std::ios::out | std::ios::binary);
			std::cout << hNumVertices << " " << hNumEdges << std::endl;
			file.write(reinterpret_cast<const char*>(&magicString.c_str()[0]), 14);	// write the file id
			file.write(reinterpret_cast<const char*>(&desc.c_str()[0]), 58);		// write the description
			file.write(reinterpret_cast<const char*>(&hNumVertices), sizeof(int));	// write #vert.
			file.write(reinterpret_cast<const char*>(&hNumEdges), sizeof(int));		// write #edges
			file.close();
		}

		// output the network as a string
		std::string str() {
			std::stringstream ss;
			size_t nv = V.size();
			size_t ne = E.size();
			ss << "Node (" << nv << ")--------" << std::endl;
			for (size_t i = 0; i < nv; i++)
				ss << "\t" << i << V[i].str() << std::endl;
			ss << "Edge (" << ne << ")--------" << std::endl;
			for (size_t i = 0; i < ne; i++)
				ss << "\t" << i << E[i].str() << std::endl;

			return ss.str();
		}

		// get a string of edges
		std::string strTxt(std::vector<std::vector<T> > p) {
			std::stringstream ss;
			std::stringstream oss;
			size_t num = p.size();
			for (size_t i = 0; i < p; i++) {
				ss.str(std::string());
				for (size_t j = 0; j < 3; j++)
					ss << p[i][j];
				ss << "\n";
			}

			return ss.str();
		}

		// removes specified character from string
		void removeCharsFromString(std::string &str, char* charsToRemove) {
			for (size_t i = 0; i < strlen(charsToRemove); i++)
				str.erase((remove(str.begin(), str.end(), charsToRemove[i])), str.end());
		}

		// exports network to txt file
		void to_txt(std::string filename) {
			
			std::ofstream ofs(filename.c_str(), std::ofstream::out | std::ofstream::app);

			ofs << (E.size()).str() << "\n";
			for (size_t i = 0; i < E.size(); i++) {
				std::string str;
				ofs << (E[i].size()).str() << "\n";
				str = E[i].strTxt();
				ofs << str << "\n";
			}
			for (size_t i = 0; i < V.size(); i++) {
				std::string str;
				str = V[i].str();
				char temp[4] = "[],";
				removeCharsFromString(str, temp);
				ofs << str << "\n";
			}
			ofs.close();
		}


		/// advanced operations
		// adding a fiber to current network
		// prior information: attaching fiber "f" and point "p" information, if "order" = 0 means the first point on fiber "e" is the attaching one (others mean the last point)
		// "add" = 0 means replace the point on fiber "e" which is to be attached to the point on current fiber, "add" = 1 means add that one. Default "add" = 0
		// ********** we don't accept that one fiber has only 3 points on it, reorder if this happens **********
		void add_fiber(edge e, size_t f, size_t p, size_t order, size_t add = 0) {
			size_t num = E[f].size();		// get the number of points on this fiber
			size_t num1 = p + 1;			// first "half"
			size_t num2 = num - p;			// second "half"
			size_t id = p;					// split point on the fiber that attach to
			size_t ne = e.size();

			// if a new fiber only has points that less than 4, either add or change (default) an attaching point
			if (num1 < 4 && num > 4)
				id = 0;
			else if (num2 < 4 && num > 4)
				id = num - 1;

			// check to see whether it needs to replace/add the joint point
			if (add == 0) {				// change the point
				if (order == 0) {
					e[0] = E[f][id];
					e.set_r(0, r(f, id));
				}
				else {
					e[ne - 1] = E[f][id];
					e.set_r(ne - 1, r(f, id));
				}
			}
			else {						// add a point if necessary
				if (order == 0) {
					if (e[0] == E[f][id])	// if they share the same joint point, make sure radii are consistent
						e.set_r(0, r(f, id));
					else
						e.insert(0, E[f][id], r(f, id));
				}
				else {
					if (e[ne - 1] == E[f][id])
						e.set_r(ne - 1, r(f, id));
					else
						e.insert(ne - 1, E[f][id], r(f, id));
				}
			}

			std::vector<edge> tmp_edge = E[f].split(id);			// check and split
																	// current one hasn't been splitted
			if (tmp_edge.size() == 1) {
				if (id == 0) {		// stitch location is the starting point of current edge
					if (V[E[f].v[0]].e[0].size() + V[E[f].v[0]].e[1].size() > 1) {	// branching point
						if (order == 0) {
							V[E[f].v[0]].e[0].push_back(E.size());
							edge new_edge(e);
							new_edge.v[0] = E[f].v[0];	// set the starting and ending points for the new edge
							new_edge.v[1] = V.size();
							E.push_back(new_edge);
							vertex new_vertex = e[e.size() - 1];
							new_vertex.e[1].push_back(E.size());	// set the incoming edge for the new point
							V.push_back(new_vertex);
						}
						else {
							V[E[f].v[0]].e[1].push_back(E.size());
							edge new_edge(e);
							new_edge.v[0] = V.size();	// set the starting and ending points for the new edge
							new_edge.v[1] = E[f].v[0];
							E.push_back(new_edge);
							vertex new_vertex = e[e.size() - 1];
							new_vertex.e[0].push_back(E.size());	// set the outgoing edge for the new point
							V.push_back(new_vertex);
						}
					}
					else {								// not branching point
						size_t k = E[f].v[0];			// get the index of the starting point on current edge
						vertex new_vertex;
						edge new_edge;
						if (order == 0) {
							new_vertex = e[e.size() - 1];
							new_edge = E[f].concatenate(e, 0, 0);
						}
						else {
							new_vertex = e[0];
							new_edge = E[f].concatenate(e, 0, e.size() - 1);
						}
						new_vertex.e[0].push_back(f);
						new_edge.v[1] = E[f].v[1];		// set starting and ending points for the new concatenated edge
						new_edge.v[0] = E[f].v[0];
						V[k] = new_vertex;
						E[f] = new_edge;
					}
				}
				else {			// stitch location is the ending point of current edge
					if (V[E[f].v[1]].e[0].size() + V[E[f].v[1]].e[1].size() > 1) {	// branching point
						if (order == 0) {
							V[E[f].v[1]].e[0].push_back(E.size());
							edge new_edge(e);
							new_edge.v[0] = E[f].v[1];	// set the starting and ending points for the new edge
							new_edge.v[1] = V.size();
							E.push_back(new_edge);
							vertex new_vertex = e[e.size() - 1];
							new_vertex.e[1].push_back(E.size() - 1);	// set the incoming edge for the new point
							V.push_back(new_vertex);
						}
						else {
							V[E[f].v[1]].e[1].push_back(E.size());
							edge new_edge(e);
							new_edge.v[0] = V.size();	// set the starting and ending points for the new edge
							new_edge.v[1] = E[f].v[1];
							E.push_back(new_edge);
							vertex new_vertex = e[e.size() - 1];
							new_vertex.e[0].push_back(E.size() - 1);	// set the outgoing edge for the new point
							V.push_back(new_vertex);
						}
					}
					else {								// not branching point
						size_t k = E[f].v[1];			// get the index of the ending point on current edge
						vertex new_vertex;
						edge new_edge;
						if (order == 0) {
							new_vertex = e[e.size() - 1];
							new_edge = E[f].concatenate(e, num - 1, 0);
						}
						else {
							new_vertex = e[0];
							new_edge = E[f].concatenate(e, num - 1, e.size() - 1);
						}
						new_vertex.e[1].push_back(f);
						new_edge.v[1] = E[f].v[1];	// set starting and ending points for the new concatenated edge
						new_edge.v[0] = E[f].v[0];
						V[k] = new_vertex;
						E[f] = new_edge;
					}
				}
			}
			// current one has been splitted
			else {
				vertex new_vertex = E[f][id];
				V.push_back(new_vertex);
				tmp_edge[0].v[0] = E[f].v[0];
				tmp_edge[0].v[1] = V.size() - 1;		// set the ending point of the first half edge
				tmp_edge[1].v[0] = V.size() - 1;		// set the starting point of the second half edge
				tmp_edge[1].v[1] = E[f].v[1];
				edge tmp(E[f]);
				E[f] = tmp_edge[0];						// replace current edge by the first half edge
				E.push_back(tmp_edge[1]);
				V[V.size() - 1].e[0].push_back(E.size() - 1);			// set the incoming and outgoing edges for the splitted point
				V[V.size() - 1].e[1].push_back(f);						// push "f" fiber as an incoming edge for the splitted point
				for (size_t i = 0; i < V[tmp.v[1]].e[1].size(); i++)	// set the incoming edge for the original ending vertex
					if (V[tmp.v[1]].e[1][i] == f)
						V[tmp.v[1]].e[1][i] = E.size() - 1;

				if (order == 0) {
					e.v[0] = V.size() - 1;				// set the starting and ending points for the new edge
					e.v[1] = V.size();
					V[V.size() - 1].e[0].push_back(E.size());	// we assume "flow" flows from starting point to ending point!
					new_vertex = e[e.size() - 1];		// get the ending point on the new edge
					E.push_back(e);
					V.push_back(new_vertex);
					V[V.size() - 1].e[1].push_back(E.size() - 1);
				}
				else {
					e.v[0] = V.size();					// set the starting and ending points for the new edge
					e.v[1] = V.size() - 1;
					V[V.size() - 1].e[1].push_back(E.size());
					new_vertex = e[0];					// get the ending point on the new edge
					E.push_back(e);
					V.push_back(new_vertex);
					V[V.size() - 1].e[0].push_back(E.size() - 1);
				}
			}
		}

		// THIS IS FOR PAVEL
		// @param "e" is the edge that is to be stitched
		// @param "f" is the index of edge that is to be stiched to
		// @param "order" means the first/last one on edge "e" to stitch
		void stitch(edge e, size_t f, size_t order) {
			network<T> A = (*this);			// make a copy of current network

			if (f >= A.edges())
				std::cout << "Current network doesn't have the "<<f<<"th fiber." << std::endl;
			else {
				T* query_point = new T[3];
				for (size_t k = 0; k < 3; k++)
					query_point[k] = e[0][k];	// we assume the first one is the one to be stitched

				size_t num = A.E[f].size();		// get the number of points on edge "f"
				T* reference_point = (T*)malloc(sizeof(T) * num * 3);
				A.E[f].edge_to_array(reference_point);
				size_t max_tree_level = 3;

				stim::kdtree<T, 3> kdt;					// initialize a tree
				kdt.create(reference_point, num, max_tree_level);		// build a tree

				T* dist = new T[1];
				size_t* nnIdx = new size_t[1];

#ifdef __CUDACC__
				kdt.search(query_point, 1, nnIdx, dist);	// search for nearest neighbor
#else
				kdt.cpu_search(query_point, 1, nnIdx, dist);// search for nearest neighbor
#endif
				add_fiber(e, f, nnIdx[0], order, 1);

				free(reference_point);
				delete(dist);
				delete(nnIdx);
			}	
		}

		// split current network at "idx" location on edge "f"
		network<T> split(size_t f, size_t idx) {
			size_t num = E.size();

			if (f >= num) {				// outside vector size
			}
			else {
				if (idx <= 0 || idx >= num - 1) {	// can't split at this position
				}
				else {
					std::vector<edge> list;			// a list of edges
					list = E[f].split(idx);			// split in tems of edges
													// first segment replaces original one
					edge new_edge = list[0];		// new edge
					edge tmp(E[f]);					// temp edge
					new_edge.v[0] = E[f].v[0];		// copy starting point
					new_edge.v[1] = V.size();		// set ending point
					E[f] = new_edge;				// replacement
					vertex new_vertex(new_edge[idx]);
					new_vertex.e[1].push_back(f);			// incoming edge for the new vertex
					new_vertex.e[0].push_back(E.size());	// outgoing edge for the new vertex

															// second segment gets newly push back
					new_edge = list[1];				// new edge
					new_edge.v[1] = tmp.v[1];		// copy ending point
					new_edge.v[0] = V.size();		// set starting point
					size_t n = V[tmp.v[1]].e[1].size();
					for (size_t i = 0; i < n; i++) {
						if (V[tmp.v[1]].e[1][i] == f) {
							V[tmp.v[1]].e[1][i] = E.size();
							break;
						}
					}

					V.push_back(new_vertex);
					E.push_back(new_edge);
				}
			}

			return (*this);
		}

		// resample current network
		network<T> resample(T spacing) {
			stim::network<T> result;

			result.V = V;	// copy vertices
			size_t num = E.size();	// get the number of edges
			result.E.resize(num);

			for (size_t i = 0; i < num; i++)
				result.E[i] = E[i].resample(spacing);

			return result;
		}

		// copy the point cload representing the centerline for the network into an array
		void centerline_cloud(T* dst) {
			size_t p;				// store the current edge point
			size_t P;				// store the number of points in an edge
			size_t t = 0;			// index in to the output array of points
			size_t num = E.size();
			for (size_t i = 0; i < num; i++) {
				P = E[i].size();
				for (p = 0; p < P; p++) {
					dst[t * 3 + 0] = E[i][p][0];
					dst[t * 3 + 1] = E[i][p][1];
					dst[t * 3 + 2] = E[i][p][2];
					t++;
				}
			}
		}

		// subdivide current network and represent as a explicit undirected graph
		network<T> to_graph() {
			std::vector<size_t> OI;			// a list of original vertex index
			std::vector<size_t> NI;			// a list of new vertex index
			std::vector<edge> nE;			// a list of new edge
			std::vector<vertex> nV;			// a list of new vector
			size_t id = 0;					// temp vertex index
			size_t num = E.size();			// number of edges in original network

			for (size_t i = 0; i < num; i++) {		// for every edge
				if (E[i].size() == 2) {				// *case 1* -> unsubdividable
					stim::centerline<T> line;		// create a centerline object for casting
					for (size_t j = 0; j < 2; j++)
						line.push_back(E[i][j]);	// copy points to the new centerline
				
					edge new_edge(line);			// construct a fiber

					for (size_t j = 0; j < 2; j++) {// deal with these two points
						vertex new_vertex = new_edge[j];	// copy vertex
						id = E[i].v[j];				// get the corresponding starting/ending point index in original vertex list
						std::vector<size_t>::iterator pos = std::find(OI.begin(), OI.end(), id);	// search this index through the list of new vertex index and see whether it appears
						if (pos == OI.end()) {		// new vertex
							OI.push_back(id);		// copy the original vertex index
							NI.push_back(nV.size());// push the new vertex index to the corresponding location

							new_vertex.e[j].push_back(nE.size());	// set the outgoing/incoming edge for the new vertex
							new_vertex.D++;			// increase degree
							new_edge.v[j] = nV.size();// set the starting/ending vertex for the new edge
							nV.push_back(new_vertex);
						}
						else {						// not a new vertex
							auto d = std::distance(OI.begin(), pos);
							new_edge.v[j] = NI[d];	// OI and NI are corresponding!
							nV[NI[d]].e[j].push_back(nE.size());
							nV[NI[d]].D++;			// increase degree
						}
					}

					nE.push_back(new_edge);

					// copy radii information
					nE[nE.size() - 1].set_r(0, E[i].r(0));
					nE[nE.size() - 1].set_r(1, E[i].r(1));
				}

				else {				// *case 2* -> subdividable
					for (size_t j = 0; j < E[i].size() - 1; j++) {
						stim::centerline<T> line;		// create a centerline object for casting
						for (size_t k = 0; k < 2; k++)
							line.push_back(E[i][j + k]);// copy only two points to the new centerline

						edge new_edge(line);			// construct a fiber

						if (j == 0) {					// first segment that contains original starting point
							vertex new_vertex = new_edge[0];
							id = E[i].v[0];				// get its original index
							std::vector<size_t>::iterator pos = std::find(OI.begin(), OI.end(), id);
							if (pos == OI.end()) {		// new vertex
								OI.push_back(id);
								NI.push_back(nV.size());

								new_vertex.e[0].push_back(nE.size());
								new_vertex.D++;			// increase degree
								new_edge.v[0] = nV.size();
								nV.push_back(new_vertex);
							}
							else {						// not a new vertex
								auto d = std::distance(OI.begin(), pos);
								new_edge.v[0] = NI[d];	// OI and NI are corresponding!
								nV[NI[d]].e[0].push_back(nE.size());
								nV[NI[d]].D++;			// increase degree
							}

							new_vertex = new_edge[1];	// get the internal point
							new_vertex.e[1].push_back(nE.size());
							new_vertex.D++;				// increase degree
							new_edge.v[1] = nV.size();
							nV.push_back(new_vertex);
							nE.push_back(new_edge);
						}

						else if (j == E[i].size() - 2) {// last segment that contains original ending point
							vertex new_vertex = new_edge[1];
							nV[nV.size() - 1].e[0].push_back(nE.size());
							nV[nV.size() - 1].D++;		// increase degree
							new_edge.v[0] = nV.size() - 1;

							id = E[i].v[1];				// get ending vertex index
							std::vector<size_t>::iterator pos = std::find(OI.begin(), OI.end(), id);
							if (pos == OI.end()) {		// new vertex
								OI.push_back(id);
								NI.push_back(nV.size());

								new_vertex.e[1].push_back(nE.size());
								new_vertex.D++;			// increase degree
								new_edge.v[1] = nV.size();
								nV.push_back(new_vertex);
							}
							else {						// not a new vertex
								auto d = std::distance(OI.begin(), pos);
								new_edge.v[1] = NI[d];
								nV[NI[d]].e[1].push_back(nE.size());
								nV[NI[d]].D++;			// increase degree
							}

							nE.push_back(new_edge);
						}

						else {							// internal segments
							vertex new_vertex = new_edge[1];

							// the first point is not a new point, but the second point is a new point
							nV[nV.size() - 1].e[0].push_back(nE.size());
							nV[nV.size() - 1].D++;		// increase degree
							new_vertex.e[1].push_back(nE.size());
							new_vertex.D++;				// increase degree
							new_edge.v[0] = nV.size() - 1;
							new_edge.v[1] = nV.size();
							nV.push_back(new_vertex);
							nE.push_back(new_edge);
						}

						// copy radii information
						nE[nE.size() - 1].set_r(0, E[i].r(j));
						nE[nE.size() - 1].set_r(1, E[i].r(j + 1));
					}
				}
			}

			stim::network<T> result(nE, nV);

			return result;
		}

		// this function compares two networks and returns the percentage of the current network that is missing from "A"
		// @param "A" is the network to compare to - the field is generated for A
		// @param "sigma" is the user-defined tolerance value - smaller values provide a stricter comparison
		// @param "device" is the GPU device to use - default no GPU provided
		network<T> compare(network<T> A, T sigma, int device = -1) {
			network<T> R;		// generate a network storing the result of comparison
			R = (*this);		// initialize to current network

			size_t num = A.total_points();
			T* c = (T*)malloc(sizeof(T) * num * 3);			// allocate memory for the centerline of A

			A.copy_to_array(c);				// copy points in A to a 1D array

			size_t max_tree_level = 3;		// set max tree level parameter to 3

#ifdef __CUDACC__
			cudaSetDevice(device);
			stim::kdtree<T, 3> kdt;					// initialize a tree object

			kdt.create(c, num, max_tree_level);		// build tree

			for (size_t i = 0; i < R.E.size(); i++) {
				size_t n = R.E[i].size();			// the number of points in current edge
				T* query_point = new T[3 * n];		// allocate memory for points
				T* m1 = new T[n];					// allocate memory for metrics
				T* dists = new T[n];				// allocate memory for distances
				size_t* nnIdx = new size_t[n];		// allocate memory for indices

				T* d_dists;
				T* d_m1;
				cudaMalloc((void**)&d_dists, n * sizeof(T));
				cudaMalloc((void**)&d_m1, n * sizeof(T));

				edge_to_array(query_point, R.E[i]);
				kdt.search(query_point, n, nnIdx, dists);

				cudaMemcpy(d_dists, dists, n * sizeof(T), cudaMemcpyHostToDevice);			// copy dists from host to device

																							// configuration parameters
				size_t threads = (1024 > n) ? n : 1024;
				size_t blocks = n / threads + (n % threads) ? 1 : 0;

				find_metric << <blocks, threads >> >(d_m1, n, d_dists, sigma);		// calculate the metric value based on the distance

				cudaMemcpy(m1, d_m1, n * sizeof(T), cudaMemcpyDeviceToHost);

				for (size_t j = 0; j < n; j++) {
					R.E[i].set_r(j, m1[j]);
				}
			}

#else		// if there is any GPU device, use CPU - much slower
			stim::kdtree<T, 3> kdt;
			kdt.create(c, num, max_tree_level);

			for (size_t i = 0; i < R.E.size(); i++) {			// for each edge in A

				size_t n = R.E[i].size();						// the number of points in current edge
				T* query = new T[3 * n];
				T* m1 = new T[n];
				T* dists = new T[n];
				size_t* nnIdx = new size_t[n];

				edge_to_array(query, R.E[i]);

				kdt.cpu_search(query, n, nnIdx, dists);			// find the distance between A and the current network

				for (size_t j = 0; j < n; j++) {
					m1[j] = 1.0f - gaussian(dists[j], sigma);	// calculate the metric value based on the distance
					R.E[i].set_r(j, m1[j]);						// set the error for the second point in the segment
				}
			}
#endif
			return R;
		}

		// this function compares two splitted networks to yield a mapping relationship between them according to nearest neighbor principle
		// @param "B" is the template network
		// @param "C" is the mapping relationship: C[e1] = _e1 means edge "e1" in current network is mapped to edge "_e1" in "B"
		// @param "device" is the GPU device that user want to use
		// @param "threshold" is to control mapping tolerance (threshold)
		void mapping(network<T> B, std::vector<size_t> &C, T threshold, int device = -1) {
			network<T> A;			// generate a network storing the result of the comparison
			A = (*this);

			size_t nA = A.E.size();	// the number of edges in A
			size_t nB = B.E.size();	// the number of edges in B

			C.resize(A.E.size());

			size_t num = B.total_points();			// set the number of points
			T* c = (T*)malloc(sizeof(T) * num * 3);

			B.copy_to_array(c);

			size_t max_tree_level = 3;

#ifdef __CUDACC__
			cudaSetDevice(device);
			stim::kdtree<T, 3> kdt;				// initialize a tree

			kdt.create(c, num, max_tree_level);		// build a tree

			T M = 0.0f;
			for (size_t i = 0; i < nA; i++) {
				M = A.ar(i);			// get the average metric of edge "i"
				if (M > threshold)
					C[i] = UINT_MAX;	// set to MAX
				else {
					T* query_point = new T[3];
					T* dist = new T[1];
					size_t* nnIdx = new size_t[1];

					for (size_t k = 0; k < 3; k++)
						query_point[k] = A.E[i][A.E[i].size() / 2][k];		// search by the middle one, risky?
					kdt.search(query_point, 1, nnIdx, dist);

					size_t id = 0;
					size_t sum = 0;
					for (size_t j = 0; j < nB; j++) {		// low_band
						sum += B.E[j].size();
						if (nnIdx[0] < sum) {
							C[i] = id;
							break;
						}
						id++;
					}
				}
			}

#else
			stim::kdtree<T, 3> kdt;
			kdt.create(c, num, max_tree_level);
			T* dist = new T[1];
			size_t* nnIdx = new size_t[1];

			stim::vec3<T> p;
			T* query_point = new T[3];

			T M = 0.0f;
			for (size_t i = 0; i < nA; i++) {
				M = A.ar(i);
				if (M > threshold)
					C[i] = UINT_MAX;
				else {
					p = A.E[i][A.E[i].size() / 2];
					for (size_t k = 0; k < 3; k++)
						query_point[k] = p[k];
					kdt.cpu_search(query_point, 1, nnIdx, dist);

					size_t id = 0;
					size_t sum = 0;
					for (size_t j = 0; j < nB; j++) {
						sum += B.E[j].size();
						if (nnIdx[0] < sum) {
							C[i] = id;
							break;
						}
						id++;
					}
				}
			}
#endif
		}
	};

}
#endif