#ifndef JACK_GL_NETWORK
#define JACK_GL_NETWORK

#include"network.h"
#include<GL/glut.h>
#include<stim/visualization/aaboundingbox.h>

namespace jack {
	
	template<typename T>
	class gl_network : public jack::network<T> {
	private:
		std::vector<T> ecolor;				// colormap for edges
		std::vector<T> vcolor;				// colormap for vertices
		GLint subdivision;					// rendering subdivision

		/// basic geometry rendering
		// main sphere rendering function
		void sphere(T x, T y, T z, T radius) {
			glPushMatrix();
			glTranslatef((GLfloat)x, (GLfloat)y, (GLfloat)z);
			glutSolidSphere((double)radius, subdivision, subdivision);
			glPopMatrix();
		}
		// rendering sphere using glut function
		void draw_sphere(T x, T y, T z, T radius, T alpha = 1.0f) {
			if (alpha != 1.0f) {
				glEnable(GL_BLEND);									// enable color blend
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
				glDisable(GL_DEPTH_TEST);							// disable depth buffer
				glColor4f(vcolor[0], vcolor[1], vcolor[2], alpha);	// set color
				sphere(x, y, z, radius, subdivision);
				glDisable(GL_BLEND);								// disable color blend
				glEnable(GL_DEPTH_TEST);							// enbale depth buffer again
			}
			else {
				glColor3f(vcolor[0], vcolor[1], vcolor[2]);
				sphere(x, y, z, radius, subdivision);
			}
		}
		// rendering sphere using quads
		void draw_sphere(T x, T y, T z, T radius, T alpha = 1.0f) {
			GLint stack = subdivision;
			GLint slice = subdivision;
			
			if (alpha != 1.0f) {
				glEnable(GL_BLEND);									// enable color blend
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
				glDisable(GL_DEPTH_TEST);							// disable depth buffer
			}
			glColor4f(vcolor[0], vcolor[1], vcolor[2], alpha);		// set color

			T step_z = stim::PI / slice;					// step angle along z-axis
			T step_xy = 2 * stim::PI / stack;				// step angle in xy-plane
			T xx[4], yy[4], zz[4];							// store coordinates

			T angle_z = 0.0;								// start angle
			T angle_xy = 0.0;

			glBegin(GL_QUADS);
			for (unsigned i = 0; i < slice; i++) {			// around the z-axis
				angle_z = i * step_z;						// step step_z each time

				for (unsigned j = 0; j < stack; j++) {		// along the z-axis
					angle_xy = j * step_xy;					// step step_xy each time, draw floor by floor

					xx[0] = radius * std::sin(angle_z) * std::cos(angle_xy);	// four vertices
					yy[0] = radius * std::sin(angle_z) * std::sin(angle_xy);
					zz[0] = radius * std::cos(angle_z);

					xx[1] = radius * std::sin(angle_z + step_z) * std::cos(angle_xy);
					yy[1] = radius * std::sin(angle_z + step_z) * std::sin(angle_xy);
					zz[1] = radius * std::cos(angle_z + step_z);

					xx[2] = radius * std::sin(angle_z + step_z) * std::cos(angle_xy + step_xy);
					yy[2] = radius * std::sin(angle_z + step_z) * std::sin(angle_xy + step_xy);
					zz[2] = radius * std::cos(angle_z + step_z);

					xx[3] = radius * std::sin(angle_z) * std::cos(angle_xy + step_xy);
					yy[3] = radius * std::sin(angle_z) * std::sin(angle_xy + step_xy);
					zz[3] = radius * std::cos(angle_z);

					for (unsigned k = 0; k < 4; k++) {
						glVertex3f(x + xx[k], y + yy[k], z + zz[k]);			// draw the floor plane
					}
				}
			}
			glEnd();

			if (alpha != 1.0f) {
				glDisable(GL_BLEND);								// disable color blend
				glEnable(GL_DEPTH_TEST);							// enbale depth buffer again
			}
		}

		// render edge as cylinders
		void draw_cylinder(T scale = 1.0f) {
			stim::circle<T> C1, C2;					// temp circles
			std::vector<stim::vec3<T> > Cp1, Cp2;	// temp lists for storing points on those circles
			T r1, r2;								// temp radii

			size_t num_edge = E.size();
			size_t num_vertex = V.size();

			for (size_t i = 0; i < num_edge; i++) {			// for every edge
				glColor3f(ecolor[i * 3 + 0], ecolor[i * 3 + 1], ecolor[i * 3 + 2]);
				for (size_t j = 1; j < num_vertex; j++) {	// for every vertex except first one
					C1 = E[i].circ(j - 1);			// get the first circle plane of this segment
					C2 = E[i].circ(j);				// get the second circle plane of this segment
					r1 = E[i].r(j - 1);				// get the radius at first point
					r2 = E[i].r(j);					// get the radius at second point
					C1.set_R(scale * r1);			// rescale
					C2.set_R(scale * r2);
					Cp1 = C1.glpoints((unsigned)subdivision);	// get 20 points on first circle plane
					Cp2 = C2.glpoints((unsigned)subdivision);	// get 20 points on second circle plane

					glBegin(GL_QUAD_STRIP);
					for (size_t k = 0; k < subdivision + 1; k++) {
						glVertex3f(Cp1[k][0], Cp1[k][1], Cp1[k][2]);
						glVertex3f(Cp2[k][0], Cp2[k][1], Cp2[k][2]);
					}
					glEnd();
				}
			}
		}


	protected:
		using jack::network<T>::E;
		using jack::network<T>::T;

		GLuint dlist;

	public:

		/// constructors
		// empty constructor
		gl_network() : jack::network<T>() {
			dlist = 0;
			subdivision = 20;
		}

		gl_network(jack::network<T> N) : jack::network<T>(N) {
			dlist = 0;
			subdivision = 20;
			ecolor.resize(N.edges * 3, 0.0f);		// default black color
			vcolor.resize(N.vertices * 3, 0.0f);
		}

		// compute the smallest bounding box of current network
		aabboundingbox<T> boundingbox() {
			aaboundingbox<T> bb;			// create a bounding box object

			for (size_t i = 0; i < E.size(); i++) 			// for every edge
				for (size_t j = 0; j < E[i].size(); j++) 	// for every point on that edge
					bb.expand(E[i][j]);		// expand the bounding box to include that point

			return bb;
		}

		// change subdivision
		void change_subdivision(GLint value) {
			subdivision = value;
		}


		/// rendering functions
		// render centerline
		void centerline() {		
			size_t num = E.size();				// get the number of edges
			for (size_t i = 0; i < num; i++) {
				glColor3f(colormap[i * 3 + 0], colormap[i * 3 + 1], colormap[i * 3 + 2]);
				glBegin(GL_LINE_STRIP);
				for (size_t j = 0; j < E[i].size(); j++)
					glVertex3f(E[i][j][0], E[i][j][1], E[i][j][2]);
				glEnd();
			}
		}


		// render network as bunch of spheres and cylinders
		void network(T scale = 1.0f) {
			stim::vec3<T> v1, v2;			// temp vertices for rendering
			T r1, r2;						// temp radii for rendering

			if (!glIsList(dlist)) {					// if dlist is not a display list, create one
				dlist = glGenLists(1);				// generate a display list
				glNewList(dlist, GL_COMPILE);		// start a new display list
			
				// render every vertex as a sphere
				size_t num = V.size();
				for (size_t i = 0; i < num; i++) {
					v1 = stim::vec3<T>(V[i][0], V[i][1], V[i][2]);	// get the vertex for rendering
					r1 = (*this).network<T>::r(i);
					draw_sphere(v1[0], v1[1], v1[2], r1 * scale);
				}

				// render every edge as a cylinder
				draw_cylinder(scale);

				glEndList();						// end the display list
			}
			glCallList(dlist);
		}
	};
}

















#endif