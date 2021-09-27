#ifndef JACK_CENTERLINE_H
#define JACK_CENTERLINE_H

#include <vector>
#include <stim/math/vec3.h>

namespace stim {

	template<typename T>
	class centerline {

	private:
		size_t n;								// number of points

												// update length information at each point (distance from starting point) starting from index "start"
		void update_L(size_t start = 0) {
			L.resize(n);

			if (start == 0) {
				L[0] = 0.0f;
				start++;
			}

			stim::vec3<T> dir;					// temp direction vector for calculating length
			for (size_t i = start; i < n; i++) {
				dir = C[i] - C[i - 1];
				L[i] = L[i - 1] + dir.len();
			}
		}

	protected:
		std::vector<stim::vec3<T> > C;			// points on the centerline
		std::vector<T> L;						// stores the integrated length along the fiber

	public:

		/// constructors
		// empty constructor
		centerline() {
			n = 0;
		}

		// constructor that allocate memory
		centerline(size_t s) {
			n = s;
			C.resize(s);
			L.resize(s);

			update_L();
		}

		// constructor that constructs a centerline based on a list of points
		centerline(std::vector<stim::vec3<T> > rhs) {
			n = rhs.size();						// get the number of points
			C.resize(n);
			for (size_t i = 0; i < n; i++)
				C[i] = rhs[i];					// copy data
			update_L();
		}


		/// vector operations
		// add a new point to current centerline
		void push_back(stim::vec3<T> p) {
			C.push_back(p);
			n++;								// increase the number of points
			update_L(n - 1);
		}

		// insert a new point at specific location to current centerline
		void insert(size_t p, stim::vec3<T> v) {
			C.insert(C.begin() + p, v);									// insert a new point
			n++;
			update_L(p);
		}

		// amend a existing point at specific location to current centerline
		void amend(size_t p, stim::vec3<T> v) {
			C[p] = v;
			update_L(p);
		}

		// erase a new point at specific location on current centerline
		void erase(size_t p) {
			C.erase(C.begin() + p);										// erase a point
			n--;
			update_L(p);
		}

		// clear up all the points
		void clear() {
			C.clear();			// clear list
			n = 0;				// set number to zero
			L.clear();			// clear length information
		}

		// resize current centerline
		void resize(size_t s) {
			n = s;
			C.resize(s);
			L.resize(s);

			update_L();
		}

		// reverse the order
		stim::centerline<T> reverse() {
		
			std::vector<stim::vec3<T> > rC = C;
			std::reverse(rC.begin(), rC.end());		// reverse the points order
			stim::centerline<T> result = rC;

			return result;
		}

		// return the number of points
		size_t size() {
			return n;
		}

		// return the length
		T length() {
			return L.back();
		}

		// get the normalized direction vector at point idx (average of the incoming and outgoing directions)
		stim::vec3<T> d(size_t idx) {
			if (n <= 1) return stim::vec3<T>(0.0f, 0.0f, 0.0f);		// if there is insufficient information to calculate the direction, return null
			if (n == 2) return (C[1] - C[0]).norm();				// if there are only two points, the direction vector at both is the direction of the line segment

																	// degenerate cases at two ends
			if (idx == 0) return (C[1] - C[0]).norm();				// the first direction vector is oriented towards the first line segment
			if (idx == n - 1) return (C[n - 1] - C[n - 2]).norm();	// the last direction vector is oriented towards the last line segment

																	// all other direction vectors are the average direction of the two joined line segments
			stim::vec3<T> a = C[idx] - C[idx - 1];
			stim::vec3<T> b = C[idx + 1] - C[idx];
			stim::vec3<T> ab = a.norm() + b.norm();
			return ab.norm();
		}


		/// arithmetic operations
		// '=' operation
		centerline<T> & operator=(centerline<T> rhs) {
			L = rhs.L;
			n = rhs.n;
			C = rhs.C;

			return *this;
		}

		// "[]" operation
		stim::vec3<T> & operator[](size_t idx) {
			return C[idx];
		}


		/// advanced operation
		// stitch two centerlines if possible
		static std::vector<centerline<T> > stitch(centerline<T> c) {
		}

		// split current centerline at specific position
		std::vector<centerline<T> > split(size_t idx) {
			std::vector<centerline<T> > result;

			// won't split
			if (idx <= 0 || idx >= n - 1) {
				result.resize(1);
				result[0] = *this;				// return current centerline
			}
			// do split
			else {
				size_t n1 = idx + 1;			// vertex idx would appear twice
				size_t n2 = n - idx;

				centerline<T> tmp;				// temp centerline

				result.resize(2);

				for (size_t i = 0; i < n1; i++)	// first half
					tmp.push_back(C[i]);
				tmp.update_L();
				result[0] = tmp;
				tmp.clear();					// clear up for next computation

				for (size_t i = 0; i < n2; i++)	// second half
					tmp.push_back(C[i + idx]);
				tmp.update_L();
				result[1] = tmp;
			}

			return result;
		}

		// resample current centerline
		centerline<T> resample(T spacing) {

			stim::vec3<T> dir;				// direction vector
			stim::vec3<T> tmp;				// intermiate point to be added
			stim::vec3<T> p1;				// starting point
			stim::vec3<T> p2;				// ending point

			centerline<T> result;

			for (size_t i = 0; i < n - 1; i++) {
				p1 = C[i];
				p2 = C[i + 1];

				dir = p2 - p1;				// compute the direction of current segment
				T seg_len = dir.len();

				if (seg_len > spacing) {	// current segment can be sampled
					for (T step = 0.0f; step < seg_len; step += spacing) {
						tmp = p1 + dir * (step / seg_len);		// add new point
						result.push_back(tmp);
					}
				}
				else
					result.push_back(p1);	// push back starting point
			}
			result.push_back(p2);			// push back ending point

			return result;
		}
	};
}

#endif