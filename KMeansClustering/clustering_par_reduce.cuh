#pragma once
#include "cuda_runtime.h"
#include "points_data.h"


// obliczanie lokalnych sum dla ka¿dej ze wspó³rzêdnych centrum idx-tego klastra i zapisywanie ich w tablicy w formacie
// pasuj¹cym dla funkcji reduceLocalSums, ka¿da ze wspó³rzêdnych ma swoj¹ przestrzeñ w tablicy, po której bêdzie prowadzone sumowanie
__global__ void prepReduceForIndexKernel(int* indicies, float* data, float* out_array, int* ndk, int stride, int idx);

// jednoczesne sumowanie wszystkich wspó³rzêdnych jednego z centrów klastra, ostatecznie funkcja zwraca sumy wspó³rzêdnych dla pewnego klastra
__global__ void reduceLocalSums(float* v, int stride, int size, int new_stride);

// obliczanie przestrzeni wymaganej w tablicy funkcji redukcji dla pojedynczej wspó³rzêdnej klastra 
int calculateStride(int n, int N_THREADS);

void Kmeans_par_modifiedReduce(PointsData_SOA& data, int MAX_ITERS);
