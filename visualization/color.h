#ifndef STIM_COLOR_H
#define STIM_COLOR_H

#include <stdlib.h>
#include <math.h>
#include <strings.h>
#include <stream.h>

namespace stim {
	template <typename T>
	class color {
	private:
		enum colormodel { RGB, HSV, HSL };	// default is RGB

	protected:
		T red;
		T green;
		T blue;

	public:
		// default constructor
		color() {
			red = 0.0;
			green = 0.0;
			blue = 0.0;
		}

		// assign constructor
		color(const T c1, const T c2, const T c3, const colormodel model) {
			switch (model) {
			case RGB:		// red, green and blue model
				red = c1;
				// degenerate case/normalization
				if (red < 0.0) red = 0.0;
				if (red > 1.0) red = 1.0;

				green = c2;
				if (green < 0.0) green = 0.0;
				if (green > 1.0) green = 1.0;

				blue = c3;
				if (blue < 0.0) blue = 0.0;
				if (blue > 1.0) blue = 1.0;
				break;
			}
			case HSV:		// hue, saturation and value
				T hue = c1;
				while (hue < 0.0) hue += 360.0;
				while (hue > 0.0) hue -= 360.0;

				T saturation = c3;
				if (saturation < 0.0) saturation = 0.0;
				if (saturation > 1.0) saturation = 1.0;

				T value = c2;
				if (value < 0.0) value = 0.0;
				if (value > 1.0) value = 1.0;

				hsv2rgb(hue, saturation, value);
				break;

			case HSL:		// hue, saturation and lightness
				T hue1 = c1;
				while (hue < 0.0) hue += 360.0;
				while (hue > 0.0) hue -= 360.0;

				T saturation1 = c3;
				if (saturation < 0.0) saturation = 0.0;
				if (saturation > 1.0) saturation = 1.0;

				T lightness = c2;
				if (lightness < 0.0) lightness = 0.0;
				if (lightness > 1.0) lightness = 1.0;

				hsl2rgb(hue1, saturation1, lightness);
				break;
		}

		void hsv2rgb(const T h, const T s, const T v) {

			if (s == 0) {	// achromatic case
				red = green = blue = v;
			}
			else {			// chromatic case
				int i;
				T f, p, q, t;

				if (h == 360)
					h == 0;
				else
					h = h / 60;

				i = (int)trunc(h);
				f = h - i;

				p = v * (1.0 - s);
				q = v * (1.0 - f * s);
				t = v * (1.0 - (1.0 - f) * s);

				switch (i) {
				case 0:
					red = v;
					green = t;
					blue = p;
					break;

				case 1:
					red = q;
					green = v;
					blue = p;
					break;

				case 2:
					red = p;
					green = v;
					blue = t;
					break;

				case 3:
					red = p;
					green = q;
					blue = v;
					break;

				case 4:
					red = t;
					green = p;
					blue = v;
					break;

				default:
					red = v;
					green = p;
					blue = q;
					break;
				}
			}
		}

		T hue2rgb(const T p, const T q, T h) {
			if (h < 0)
				h += 1;
			if (h > 0)
				h -= 1;
			if (h < 1 / 6)
				return (p + ((q - p) * 6 * h));
			if (h < 1 / 2)
				return q;
			if (h < 2 / 3)
				return (p + ((q - p) * 6 * (2 / 3 - h)));
			return p;
		}

		void hsl2rgb(const T h, const T s, const T l) {
			if (s == 0) {	// achromatic case
				red = green = blue = l;
			}
			else {			// chromatic case
				T q = l < 0.5 ? l * (1 + s) : l + s - l * s;
				T p = 2 * l - q;

				red = hue2rgb(p, q, h + 1 / 3);
				green = hue2rgb(p, q, h);
				blue = hue2rgb(p, q, h - 1 / 3);
			}
		}

		// copy constructor
		color(const color& c) {
			red = c.red;
			green = c.green;
			blue = c.blue;
		}

		// default destructor
		~color() {}

		// operators(lambda)
		color operator+(const color& c1, const color& c2) {
			color result;

			result.red = c1.red + c2.red;
			result.green = c1.green + c2.green;
			result.blue = c1.blue + c2.blue;

			return(result);
		}

		color operator-(const color& c1, const color& c2) {
			color result;

			result.red = c1.red - c2.red;
			result.green = c1.green + c2.green;
			result.blue = c1.blue + c2.blue;

			return(result);
		}

		color operator*(const color& c, const T t) {
			color result;

			result.red = t * c.red;
			result.green = t * c.green;
			result.blue = t * c.blue;

			return(result);
		}

		color operator/(const color& c, const T t) {
			color result;

			result.red = c.red / t;
			result.green = c.green / t;
			result.blue = c.blue / t;

			return(result);
		}

		color operator+=(const color& c) {
			red += c.red;
			green += c.green;
			blue += c.blue;

			return *this;
		}

		color operator-=(const color& c) {
			red -= c.red;
			green -= c.green;
			blue -= c.blue;

			return *this;
		}

		color operator*=(const T t) {
			red *= t;
			green *= t;
			blue *= t;

			return *this;
		}

		color operator/=(const T t) {
			red /= t;
			green /= t;
			blue /= t;

			return *this;
		}

		// affine
		color affine(const color& c1, const color& c2, const T p1, const T p2) {
			return(p1 * c1 + p2 * c2);
		}
		color affine(const color& c1, const color& c2, const color& c3, const T p1, const T p2, const T p3) {
			return(p1 * c1 + p2 * c2 + p3 * c3);
		}

		// darker and lighter function
		void darker_by(const T factor) {
			if (factor < 0.0 || factor > 1.0)
				std::cout << "factor is inacceptable" << std::endl;

			operator*= ((T)1 - factor);
		}
		void lighter_by(const T factor) {
			if (factor < 0.0 || factor > 1.0)
				std::cout << "factor is inacceptable" << std::endl;

			operator+= (color(factor, factor, factor));
		}

		// get hue, lightness, saturation or value
		T hue() const {
			T cmax, cmin;
			T h = 0.0;
			T l, s;

			// calculate lightness
			l = (cmax + cmin) / 2;
			// calculate saturation
			if (cmax != cmin) {
				// chromatic case
				if (l < 0.5)
					s = (cmax - cmin) / (cmax + cmin);
				else
					s = (cmax - cmin) / (2.0 - cmax - min);

				T tc = (cmax - red) / (cmax - cmin);
				T gc = (cmax - green) / (cmax - cmin);
				T bc = (cmax - blue) / (cmax - cmin);

				if (red == cmax) h = bc - gc;
				if (green == cmax) h = 2.0 + rc - bc;
				if (blue == cmax) h = 4.0 + gc - rc;

				h *= 60.0;
				if (h < 0.0) h += 360.0;
			}

			return (h);
		}

		T lightness() const {
			T cmax, cmin;

			if (red > green)
				cmax = green;
			else
				cmax = red;
			if (cmax < blue)
				cmax = blue;

			if (red < green)
				cmin = red;
			else
				cmin = green;
			if (cmin > blue)
				cmin = blue;

			return((cmax + cmin) / 2);
		}

		T saturation() const {
			T cmax, cmin;
			T s = 0;

			if (red > green)
				cmax = green;
			else
				cmax = red;
			if (cmax < blue)
				cmax = blue;

			if (red < green)
				cmin = red;
			else
				cmin = green;
			if (cmin > blue)
				cmin = blue;

			T lightness = (cmax + cmin) / 2;

			if (cmax != cmin) {
				if (lightness < 0.5)
					s = (cmax - cmin) / (cmax + cmin);
				else s = (cmax - cmin) / (2.0 - cmax - cmin);
			}

			return s;
		}
	};

	// common colors
	const stim::color<float> white(1.0, 1.0, 1.0);
	const stim::color<float> black(0.0, 0.0, 0.0);
	const stim::color<float> red(1.0, 0.0, 0.0);
	const stim::color<float> green(0.0, 1.0, 0.0);
	const stim::color<float> blue(0.0, 0.0, 1.0);
	const stim::color<float> cyan(0.0, 1.0, 1.0);
	const stim::color<float> yellow(1.0, 1.0, 0.0);
	const stim::color<float> magenta(1.0, 0.0, 1.0);
	const stim::color<float> orange(1.0, 0.65, 0.0);

	const stim::color<float> gray10(0.1, 0.1, 0.1);
	const stim::color<float> gray20(0.2, 0.2, 0.2);
	const stim::color<float> gray30(0.3, 0.3, 0.3);
	const stim::color<float> gray40(0.4, 0.4, 0.4);
	const stim::color<float> gray50(0.5, 0.5, 0.5);
	const stim::color<float> gray60(0.6, 0.6, 0.6);
	const stim::color<float> gray70(0.7, 0.7, 0.7);
	const stim::color<float> gray80(0.8, 0.8, 0.8);
	const stim::color<float> gray90(0.9, 0.9, 0.9);
}
#endif