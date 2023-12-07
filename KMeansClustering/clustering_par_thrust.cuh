#include "cuda_runtime.h"
#include "points_data.h"


// funkcja tworzy tablicê indeksów, gdzie ka¿da wspó³rzêdna ka¿dego centrum posiada swój w³asny indeks
// jest to celowe dzia³anie aby dopasowaæ indeksy dla danych do funkcji reduce_by_key
__global__ void createIndexMapKernel(int* indicies, int* ndk, int* out_indexmap);
void Kmeans_par_thrust(PointsData_SOA& data, int MAX_ITERS);
