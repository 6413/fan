#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <cuda.h>

#include <nvcuvid.h>

#include "device_launch_parameters.h"
//#define HGPUNV void*
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void call_kernel(cudaSurfaceObject_t surface, int, int);

int main()
{

  fan::print("cpu");

  cudaSurfaceObject_t test;

  call_kernel(test, 1, 1);
  return 0;
}