#include <cuda_runtime.h>
#include "points_data.h"

// kernel obliczaj�cy odleg�o�ci mi�dzy punktami a �rodkami centr�w
__global__ void calcDistKernel(float* data, float* clusterData, int* ndk, float* dist);

// kernel aktualizuj�cy przynale�no�� punktu do centrum
__global__ void updateIndexKernel(float* min_dist, int* ndk, int* indicies, int* delta);

int calculateNumThreads(int k);

void Kmeans_par_UpdateClusterOnCPU(PointsData_SOA& data, int MAX_ITERS);
