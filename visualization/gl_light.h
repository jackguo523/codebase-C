#ifndef STIM_GL_LIGHT
#define STIM_GL_LIGHT

#include <vector>

//OPENGL include
#include <GL/freeglut.h>
#include <GL/glut.h>

//SITM include
#include <stim/math/vec3.h>

namespace stim {

	template <typename T>
	class gl_light {

	public:
		GLenum l;				// light order number
		T* pos;					// position
		T* amb;					// ambient						
		T* dif;					// diffuse
		T* spe;					// specular

		gl_light() {					// constructor
			l = GL_LIGHT0;
			amb = (T*)malloc(4 * sizeof(T));
			dif = (T*)malloc(4 * sizeof(T));
			spe = (T*)malloc(4 * sizeof(T));
			pos = (T*)malloc(4 * sizeof(T));
		}

		void init(int Light_order_number, T* ambient, T* diffuse, T* specular, T* position) {

			if (Light_order_number < 0 || Light_order_number > 7) {
				std::cout << "Sorry, OPENGL only provide LIGHT from 0 to 7" << std::endl;
				exit(1);
			}
			glClearColor(0.0, 0.0, 0.0, 1.0);

			// find out which light the user want to use
			switch (Light_order_number) {
			case 0:
				l = GL_LIGHT0;
				break;
			case 1:
				l = GL_LIGHT1;
				break;
			case 2:
				l = GL_LIGHT2;
				break;
			case 3:
				l = GL_LIGHT3;
				break;
			case 4:
				l = GL_LIGHT4;
				break;
			case 5:
				l = GL_LIGHT5;
				break;
			case 6:
				l = GL_LIGHT6;
				break;
			case 7:
				l = GL_LIGHT7;
				break;
			}
			amb = ambient;
			dif = diffuse;
			spe = specular;
			pos = position;

			if (typeid(ambient[0]) == typeid(float)) {			// let first element of ambient to determine datatype
				glLightfv(l, GL_AMBIENT, ambient);
				glLightfv(l, GL_DIFFUSE, diffuse);
				glLightfv(l, GL_SPECULAR, specular);
				glLightfv(l, GL_POSITION, position);
			}
			else {
				//glLightiv(l, GL_AMBIENT, ambient);
				//glLightiv(l, GL_DIFFUSE, diffuse);
				//glLightiv(l, GL_SPECULAR, specular);
				//glLightiv(l, GL_POSITION, position);
			}
		}

		void light(T* attenuation) {
			if (typeid(exponent) == typeid(float)) {
				glLightf(l, GL_CONSTANT_ATTENUATION, attenuation[0]);
				glLightf(l, GL_LINEAR_ATTENUATION, attenuation[1]);
				glLightf(l, GL_QUADRATIC_ATTENUATION, attenuation[2]);
			}
			else {
				glLighti(l, GL_CONSTANT_ATTENUATION, attenuation[0]);
				glLighti(l, GL_LINEAR_ATTENUATION, attenuation[1]);
				glLighti(l, GL_QUADRATIC_ATTENUATION, attenuation[2]);
			}
		}

		void light(T* direction, T* attenuation, T exponent = (T)0, T angle = (T)0) {
			if (typeid(exponent) == typeid(float)) {
				glLightfv(l, GL_SPOT_DIRECTION, direction);
				glLightf(l, GL_SPOT_EXPONENT, exponent);
				glLightf(l, GL_SPOT_CUTOFF, angle);
				glLightf(l, GL_CONSTANT_ATTENUATION, attenuation[0]);
				glLightf(l, GL_LINEAR_ATTENUATION, attenuation[1]);
				glLightf(l, GL_QUADRATIC_ATTENUATION, attenuation[2]);
			}
			else {
				glLightiv(l, GL_SPOT_DIRECTION, direction);
				glLighti(l, GL_SPOT_EXPONENT, exponent);
				glLighti(l, GL_SPOT_CUTOFF, angle);
				glLighti(l, GL_CONSTANT_ATTENUATION, attenuation[0]);
				glLighti(l, GL_LINEAR_ATTENUATION, attenuation[1]);
				glLighti(l, GL_QUADRATIC_ATTENUATION, attenuation[2]);
			}
		}

		void lightmodel(T local_viewer, T* global_ambient) {
			if (typeid(local_viewer) == typeid(float)) {
				glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
				glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, local_viewer);
			}
			else {
				glLightModeliv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
				glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, local_viewer);
			}
		}
	};
}
#endif