#pragma once

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointsData.h"

void Kmeans_par_AtomicAddNewCluster(PointsData_FlatArray& data, int kk);
void Kmeans_par_UpdateClusterOnCPU(PointsData_FlatArray& data, int kk);
void Kmeans_par_NewClusterSumGPU(PointsData_FlatArray& data, int kk);
void Kmeans_par_NewClusterSumGPUWithThrust(PointsData_FlatArray& data, int kk);
int calculateStride(int n, int N_THREADS);