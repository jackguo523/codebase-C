#ifndef STIM_VEC3_H
#define STIM_VEC3_H

#include <cmath>
#include <stim/math/constants.h>

namespace stim{

template<typename T>
class vec3{

protected:
  T dim[3];

public:
  vec3(){}          // default constructor
  vec3(T v){
    dim[0] = dim[1] = dim[2] = v;
  }
  vec3(T x, T y, T z){
    dim[0] = x;
    dim[1] = y;
    dim[2] = z;
  }

  // copy constructor
  vec3(const vec3<T> rhs){
    dim[0] = rhs.dim[0];
    dim[1] = rhs.dim[1];
    dim[2] = rhs.dim[2];
  }

  // access one dimension value
  T& operator[](size_t idx){
    return dim[idx];
  }

  // compute the Euclidean Length of the vector
  T len() const{
    T result;
    result = dim[0]*dim[0] + dim[1]*dim[1] + dim[2]*dim[2];

    return sqrt(result);
  }

  // dot product
  T dot(vec3<T> rhs) const{
    return dim[0]*rhs.dim[0] + dim[1]*rhs.dim[1] + dim[2]*rhs.dim[2];
  }

  // cross product
  vec3<T> cross(vec<T> rhs) const{
    vec3<T> result;
    result.dim[0] = dim[1]*rhs.dim[2] - rhs.dim[1]*dim[2];
    result.dim[1] = dim[2]*rhs.dim[0] - dim[0]*rhs.dim[2];
    result.dim[2] = dim[0]*rhs.dim[1] - dim[1]*rhs.dim[0];

    return result;
  }

  // convert from cartesian coordinates to cylinder coordinates(x, y, z -> r, theta, h)
  vec3<T> cart2cyl() const{
    vec3<T> cyl;    // won't consider degenerate case when r = 0
    cyl.dim[0] = sqrt(pow(dim[0], 2) + pow(dim[1], 2)); // r
    cyl.dim[1] = std::atan(dim[1]/dim[0]);                   // theta
    cyl.dim[2] = dim[2];                                // h

    return cyl;
  }

  // convert from cartesian coordinates to spherical coordinates(x, y, z -> r, phi, theta)
  vec3<T> cart2sph() const{
    vec3<T> sph;     // won't consider degenerate case when r = 0
    sph.dim[0] = sqrt(pow(dim[0], 2) + pow(dim[1], 2) + pow(dim[2], 2));  // r
    sph.dim[1] = std::atan(sqrt(pow(dim[0], 2) + pow(dim[1] , 2))/dim[2]);     // phi
    sph.dim[2] = std::atan(dim[1]/dim[0]);                                     // theta

    return sph;
  }

  // convert from cylinder coordinates to cartesian coordinates(r, theta, h -> x, y, z)
  vec3<T> cyl2cart() const{
    vec3<T> cart;
    cart.dim[0] = dim[0] * std::cos(dim[1]);
    cart.dim[1] = dim[0] * std::sin(dim[1]);
    cart.dim[2] = dim[2];

    return cart;
  }

  // convert from spherical coordinates to cartesian coordinates(r, phi, theta -> x, y, z)
  vec3<T> sph2cart() const{
    vec3<T> cart;
    cart.dim[0] = dim[0] * std::sin(dim[1]) * std::cos(dim[2]);
    cart.dim[1] = dim[0] * std::sin(dim[1]) * std::sin(dim[2]);
    cart.dim[2] = dim[0] * std::cos(dim[1]);

    return cart;
  }

  // scaling
  void scale(T Sx, T Sy, T Sz){
      dim[0] *= Sx;
      dim[1] *= Sy;
      dim[2] *= Sz;
  }

  // rotation
  void rotate_x(T deg){
    T rad = deg * stim::PI / 180;     // convert from degree to radian
    dim[1] = dim[1]*std::cos(rad) - dim[2]*std::sin(rad);
    dim[2] = dim[2]*std::sin(rad) + dim[2]*std::cos(rad);
  }
  void rotate_y(T deg){
    T rad = deg * stim::PI / 180;
    dim[0] = dim[0]*std::cos(rad) + dim[2]*std::sin(rad);
    dim[2] = -dim[0]*std::sin(rad) + dim[2]*std::cos(rad);
  }
  void rotate_z(T deg){
    T rad = deg * stim::PI / 180;
    dim[0] = dim[0]*std::cos(rad) - dim[1]*std::sin(rad);
    dim[1] = dim[0]*std::sin(rad) + dim[1]*std::cos(rad);
  }

  // shear
  void shear(T Sxy, T Sxz, T Syz, T Syx, T Szx, T Szy){
    dim[0] = dim[0] + Syx*dim[1] + Szx*dim[2];
    dim[1] = Sxy*dim[0] + dim[1] + Szy*dim[2];
    dim[2] = Sxz*dim[0] + Syz*dim[1] + dim[2];
  }

  // translation
  void translate(T dx, T dy, T dz){
    dim[0] += dx;
    dim[1] += dy;
    dim[2] += dz;
  }


  /// arithmetic addition operation

  // addition
  vec3<T> operator+(vec3<T> rhs) const{
    vec3<T> result;
    result.dim[0] = dim[0] + rhs.dim[0];
    result.dim[1] = dim[1] + rhs.dim[1];
    result.dim[2] = dim[2] + rhs.dim[2];

    return result;
  }
  vec3<T> operator+=(vec3<T> rhs) const{
    dim[0] = dim[0] + rhs.dim[0];
    dim[1] = dim[1] + rhs.dim[1];
    dim[2] = dim[2] + rhs.dim[2];

    return *this;
  }

  // subtraction
  vec3<T> operator-(vec3<T> rhs) const{
    vec3<T> result;
    result.dim[0] = dim[0] - rhs.dim[0];
    result.dim[1] = dim[1] - rhs.dim[1];
    result.dim[2] = dim[2] - rhs.dim[2];

    return result;
  }
  vec3<T> operator-=(vec<T> rhs) const{
    dim[0] = dim[0] - rhs.dim[0];
    dim[1] = dim[1] - rhs.dim[1];
    dim[2] = dim[2] - rhs.dim[2];

    return *this;
  }

  // multiplication
  T operator*(vec3<T> rhs) const{
    return dim[0]*rhs.dim[0] + dim[1]*rhs.dim[1] + dim[2]*rhs.dim[2];
  }

  // comparison
  bool operator==(vec3<T> rhs) const{
    if(dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1] && dim[2] == rhs.dim[2])
      return true;
    else
      return false;
  }

  // output the vector as a string
  std::string str() const{
    std::stringstream ss;

    ss<<"[";    // left bracket
    for(size_t i = 0; i < 3; i++){
      ss<<dim[i];
      if(i != N-1)
        ss<<", ";
    }
    ss<<"]"     // right bracket

    return ss.str();
  }

  size_t size(){
    return 3;
  }
};

#endif
