#include <cuda_runtime.h>
#include "points_data.h"

// kernel obliczaj¹cy odleg³oœci miêdzy punktami a œrodkami centrów
__global__ void calcDistKernel(float* data, float* clusterData, int* ndk, float* dist);

// kernel aktualizuj¹cy przynale¿noœæ punktu do centrum
__global__ void updateIndexKernel(float* min_dist, int* ndk, int* indicies, int* delta);

int calculateNumThreads(int k);

void Kmeans_par_UpdateClusterOnCPU(PointsData_SOA& data, int MAX_ITERS);
