#include "cuda_runtime.h"
#include "points_data.h"


// funkcja tworzy tablic� indeks�w, gdzie ka�da wsp�rz�dna ka�dego centrum posiada sw�j w�asny indeks
// jest to celowe dzia�anie aby dopasowa� indeksy dla danych do funkcji reduce_by_key
__global__ void createIndexMapKernel(int* indicies, int* ndk, int* out_indexmap);
void Kmeans_par_thrust(PointsData_SOA& data, int MAX_ITERS);
