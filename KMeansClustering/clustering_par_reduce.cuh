#pragma once
#include "cuda_runtime.h"
#include "points_data.h"


// obliczanie lokalnych sum dla ka�dej ze wsp�rz�dnych centrum idx-tego klastra i zapisywanie ich w tablicy w formacie
// pasuj�cym dla funkcji reduceLocalSums, ka�da ze wsp�rz�dnych ma swoj� przestrze� w tablicy, po kt�rej b�dzie prowadzone sumowanie
__global__ void prepReduceForIndexKernel(int* indicies, float* data, float* out_array, int* ndk, int stride, int idx);

// jednoczesne sumowanie wszystkich wsp�rz�dnych jednego z centr�w klastra, ostatecznie funkcja zwraca sumy wsp�rz�dnych dla pewnego klastra
__global__ void reduceLocalSums(float* v, int stride, int size, int new_stride);

// obliczanie przestrzeni wymaganej w tablicy funkcji redukcji dla pojedynczej wsp�rz�dnej klastra 
int calculateStride(int n, int N_THREADS);

void Kmeans_par_modifiedReduce(PointsData_SOA& data, int MAX_ITERS);
