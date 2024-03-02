#include <stdio.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void kernel(cudaSurfaceObject_t surface, int nx, int ny) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    uchar4 data = make_uchar4(x % 255,
      y % 255,
      x % 255, 255);
    surf2Dwrite(data, surface, x * sizeof(uchar4), y);
  }
}

void call_kernel(cudaSurfaceObject_t surface, int x, int y) {

  dim3 gridSize((x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (y + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);

  kernel << <gridSize, blockSize>> > (surface, x, y);
  //cudaDeviceSynchronize();
}