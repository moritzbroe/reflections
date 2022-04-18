#include <iostream>
#include <math.h>
#include <vector>
#include <Eigen/Dense>
#include <X11/Xlib.h>
#include "X11/keysym.h"
#include <unistd.h>
#include <chrono>


#define SCREEN_WIDTH 1.25
#define SCREEN_HEIGHT 0.7

// used in update_orientation
#define ANG1 0.3
#define ANG2 0.2
#define SPEED 5
#define TSTEP 0.02   // * 10^-6s
#define ANG1_RESET 1

#define EPS 0.00001   // for division by "zero"
#define MAX_REFLECTIONS 10
#define MAX_BALLS 1000    // max number of balls
#define MAX_CYLS 1000


typedef Eigen::Vector3f Vec;

using namespace std;

class Point {
public:
  float x, y, z;
  void rotate(Point axis, float amount) {
    // rotates point around axis by amount (in radians)


  }
};


class Ball {
public:
  Vec center;
  float radius;
};


class Cylinder {
public:
  // cylinder axis between base and base+dir
  Vec base;
  Vec dir;
  float radius;

};


// copied from somewhere
bool key_is_pressed(KeySym ks) {
    Display *dpy = XOpenDisplay(":1");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    bool isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}


void update_orientation(Eigen::Matrix3f *dir) {
    if (key_is_pressed(XK_Left)) {
      *dir = Eigen::AngleAxisf(-ANG1*TSTEP*M_PI, dir->col(0)) * *dir;
      *dir = Eigen::AngleAxisf(0.5*ANG1*TSTEP*M_PI, dir->col(2)) * *dir;
      *dir = Eigen::AngleAxisf(-0.4*ANG1*TSTEP*M_PI, dir->col(1)) * *dir;
    }
    if (key_is_pressed(XK_Right)) {
      *dir = Eigen::AngleAxisf(ANG1*TSTEP*M_PI, dir->col(0)) * *dir;
      *dir = Eigen::AngleAxisf(-0.5*ANG1*TSTEP*M_PI, dir->col(2)) * *dir;
      *dir = Eigen::AngleAxisf(-0.4*ANG1*TSTEP*M_PI, dir->col(1)) * *dir;
    }
    if (key_is_pressed(XK_Up)) {
      *dir = Eigen::AngleAxisf(ANG2*TSTEP*M_PI, dir->col(1)) * *dir;
    }
    if (key_is_pressed(XK_Down)) {
      *dir = Eigen::AngleAxisf(-ANG2*TSTEP*M_PI, dir->col(1)) * *dir;
    }
    float roll = dir->col(1).z();
    *dir = Eigen::AngleAxisf(-ANG1_RESET*TSTEP*roll, dir->col(0)) * *dir;
}


void update_pos(Vec *pos, Eigen::Matrix3f *dir) {
  *pos += dir->col(0)*TSTEP*SPEED;
}


Vec **init_ray_array(int height, int width) {
  Vec **res;
  cudaMallocManaged(&res, sizeof(Vec*)*height);
  for (int i = 0; i < height; ++i) {
    cudaMallocManaged(&res[i], sizeof(Vec)*width);
  }
  return res;
}


char **init_pic_array(int height, int width) {
  char **res;
  cudaMallocManaged(&res, height*sizeof(char*));
  for (int i = 0; i < height; ++i) {
    cudaMallocManaged(&res[i], (width+1)*sizeof(char));
    res[i][width] = 0;
  }
  return res;
}


__global__
void create_rays(Vec **rays, Eigen::Matrix3f *dir, int height, int width) {
  // rays go through "pixels" of "screen" in front of the plane, written into rays array
  int index = threadIdx.x;
  int stride = blockDim.x;
  Vec upper_left = dir->col(0) + SCREEN_WIDTH/2.0*dir->col(1) + SCREEN_HEIGHT/2.0*dir->col(2);
  for (int i = blockIdx.x; i < height; i += gridDim.x) {
    for (int j = index; j < width; j += stride) {
      rays[i][j] = upper_left - SCREEN_HEIGHT*(float)i/(height-1) * dir->col(2) - SCREEN_WIDTH*(float)j/(width-1) * dir->col(1);
      rays[i][j].normalize();
    }
  }
}


__device__
bool check_reflection_ball(Ball &ball, Vec &from, Vec &ray, float &min_dist_to_refl, Vec &reflection_point, Vec &reflection_plane) {
  // checks if from + t*ray intersects ball, if so checks if intersection point is closer to from than min_dist_to_refl
  // if that is the case, write reflection_point and reflection_plane, return true
  // else return false
  // math: use abc formula to solve for intersections, then use closer intersection point
  Vec tmp = from - ball.center;
  float b = 2*ray.dot(tmp);
  float c = tmp.dot(tmp) - ball.radius*ball.radius;
  float under_root = b*b - 4*c;
  if (under_root < 0) return false;   // no reflection

  float t = (- b - sqrt(under_root))/2;    // only the smaller solution is useful
  if (t < 0) return false;    // reflection behind viewer
  // t > 0, thus intersection is at pos + t*ray
  if (t < min_dist_to_refl) {
    reflection_point = from + t * ray;
    if (reflection_point.z() < 0) return false;
    reflection_plane = reflection_point - ball.center;
    min_dist_to_refl = t;
    return true;
  }
  return false;
}

__device__
bool check_reflection_cylinder(Cylinder &cyl, Vec &from, Vec &ray, float &min_dist_to_refl, Vec &reflection_point, Vec &reflection_plane) {
  // does exactly the same as check_reflection_ball, except with a cylinder
  // the math is more complicated, rough outline:
  // first check if minimal distance of ray to cylinder axis is smaller than cylinder radius
  // if so, calculate first intersection using the minimal distance point
  // then check if this intersection point is still in the bounds of the cylinder
  // not even sure if my math was correct, they sometimes look weird but most of the time ok

  // cylinder: a+tb, ray: c+sd
  Vec a, b, c, d, p, q, r, u;
  float s1, s2, t1, t2, R;

  a = cyl.base;
  b = cyl.dir;
  c = from;
  d = ray;
  R = cyl.radius;

  float dnom = b.dot(d) * b.dot(d) - b.dot(b);
  if (abs(dnom) < EPS) return false;  // cylinder axis parallel to ray
  s1 = (d.dot(c-a) * b.dot(b) - b.dot(c-a) * b.dot(d))/dnom;
  if (s1 < 0) return false; // point of minimal distance is behind viewer, thus first point of intersection as well
  t1 = (s1 * b.dot(d) + b.dot(c-a))/b.dot(b);
  p = c + s1 * d;   // point on ray with minimal distance to cylinder axis
  q = p - (a + t1 * b);        // vector from cylinder axis to p, orthogonal to cyl axis
  if (q.dot(q) >= R*R) return false;   // no reflection
  s2 = sqrt((R*R - q.dot(q))/(1-d.dot(b)/b.dot(b)));  // s2 is how far to go back from p on the ray to get to an intersection point
  if (s2 > s1) return false;  // first point of intersection is behind viewer
  if (s1-s2 > min_dist_to_refl) return false;
  reflection_point = p - s2*d;
  if (reflection_point.z() < 0) return false;
  u = reflection_point - a;
  reflection_plane = u - u.dot(b)/b.dot(b)*b;   // projected u onto cyl axis giving plane of reflection 

  // check if reflection point is in the cylinders bounds
  r = reflection_point - reflection_plane - a;  // vector from a to projection of reflection_point onto cyl axis
  if (b.x() != 0) t2 = r.x()/b.x();
  else if (b.y() != 0) t2 = r.y()/b.y();
  else if (b.z() != 0) t2 = r.z()/b.z();
  else return false;
  if (t2 >= 0 && t2 <= 1) {
    min_dist_to_refl = s1-s2;
    return true;
  }
  else return false;
}


__global__
void calc_rays(Vec *pos, Vec **rays, char **pic, Ball *balls, int num_balls, Cylinder *cyls, int num_cyls, int height, int width) {
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = blockIdx.x; i < height; i += gridDim.x) {
    for (int j = index; j < width; j += stride) {
      Vec from = *pos;
      Vec ray = rays[i][j];
      bool ever_reflected = false;
      bool reflected_now;
      Vec reflection_point;
      Vec reflection_plane;

      // do up to 10 ball reflections
      for (int k = 0; k < MAX_REFLECTIONS; ++k) {
        reflected_now = false;
        float min_dist_to_refl = 1000000;   // initialize to large distance

        // check ball reflections
        for (int ball_i = 0; ball_i < num_balls; ++ball_i) {
          if (check_reflection_ball(balls[ball_i], from, ray, min_dist_to_refl, reflection_point, reflection_plane)) {
            reflected_now = true;
          }
        }
        // and cylinder reflections
        for (int cyl_i = 0; cyl_i < num_cyls; ++cyl_i) {
          if (check_reflection_cylinder(cyls[cyl_i], from, ray, min_dist_to_refl, reflection_point, reflection_plane)) {
            reflected_now = true;
          }
        }
        if (!reflected_now) break;   // no reflection, thus no more checking necessary
        // reflect off of plane given by reflection_point and reflection_plane
        // math done on paper
        ray -= 2*ray.dot(reflection_plane)/reflection_plane.dot(reflection_plane) * reflection_plane;
        from = reflection_point + 0.001*reflection_plane;   // new start position for the ray, added a bit of refl to get out of the ball
        ever_reflected = true;
      }

      // check if it hits floor, color appropriately (using color code and string, change string to change looks)
      int color = 0;
      char code_to_char[7] = ". ;,EL";
      if (ever_reflected) color++;
      if (ray.z() < 0) {
        from -= from.z()/ray.z() * ray;
        if ((int)(floor(from.x()) + floor(from.y())) % 2 == 0) {
          // dark tiles
          color += 4;
        }
        else {
          // light tiles
          color += 2;
        }
      }
      pic[i][j] = code_to_char[color];
    }
  }
}


void draw_pic(char **pic, int height, int width) {
  cout << "\033[0;0f";
  for (int i = 0; i < height; ++i) {
    cout << pic[i];
    cout << '\n';
  }
  cout << '\n';
}


int main(int argc, char *argv[]) {
  int height, width;
  if (argc == 1) {
    // no window sizes given, defaults to 200x100
    height = 100;
    width = 200;
  }
  else {
    height = stoi(argv[2]);
    width = stoi(argv[1]);
  }

  Vec *pos;
  cudaMallocManaged(&pos, sizeof(Vec));
  *pos = {-40, -2, 8};
  // orientation is stored as a matrix:
  // from the planes perspective:
  // first column vector points forward
  // second one points to the left 
  // third one points up
  Eigen::Matrix3f *dir;
  cudaMallocManaged(&dir, sizeof(Eigen::Matrix3f));
  *dir << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  int num_balls = 0;
  int num_cyls = 0;
  Ball *balls;
  cudaMallocManaged(&balls, MAX_BALLS*sizeof(Ball));
  Cylinder *cyls;
  cudaMallocManaged(&cyls, MAX_CYLS*sizeof(Cylinder));

  // create some balls and cylinders
  for (int i = 0; i < 5; ++i) {
    cyls[num_cyls++] = (Cylinder) {(Vec) {7*i, -7, -2}, (Vec) {0, 20, 40}, 2};
    cyls[num_cyls++] = (Cylinder) {(Vec) {7*i + 3, 7, -2}, (Vec) {0, -20, 40}, 2};
  }

  cyls[num_cyls++] = (Cylinder) {(Vec) {80, 20, 0}, (Vec) {0, 0, 10}, 3};
  balls[num_balls++] = (Ball) {(Vec) {80, 20, 10}, 5};

  balls[num_balls++] = (Ball) {(Vec) {70, 70, 8}, 8};;
  balls[num_balls++] = (Ball) {(Vec) {70, 70, 22}, 6};;
  balls[num_balls++] = (Ball) {(Vec) {70, 70, 32}, 4};;

  for (int i = 1; i < 10; ++i) {
    balls[num_balls++] = (Ball) {(Vec) {20, 70, i*i}, i};;
  }
  for (int i = 1; i < 15; ++i) {
    balls[num_balls++] = (Ball) {(Vec) {-i, -30-(i+1)*(i+1), i}, i};
    cyls[num_cyls++] = (Cylinder) {(Vec) {5+i, -30-(i+1)*(i+1), -i/3}, (Vec) {0, 10+2*i, 40+8*i}, i};
  }

  // initialize stuff
  Vec **rays = init_ray_array(height, width);  // holds directions of the rays (later)
  char **pic = init_pic_array(height, width);

  int blockSize = 512;
  int numBlocks = 256;


  // main loop:
  while(1) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    update_orientation(dir);
    update_pos(pos, dir);
    create_rays<<<numBlocks, blockSize>>>(rays, dir, height, width);
    cudaDeviceSynchronize();
    calc_rays<<<numBlocks, blockSize>>>(pos, rays, pic, balls, num_balls, cyls, num_cyls, height, width);
    cudaDeviceSynchronize();
    draw_pic(pic, height, width);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    if (TSTEP*1000000 > elapsed) usleep(TSTEP*1000000 - elapsed);
  }
}