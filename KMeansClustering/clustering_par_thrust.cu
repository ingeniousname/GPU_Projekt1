#include "cudaError.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "reduce.cuh"
#include "clustering_par_thrust.cuh"
#include "clustering_par_common.cuh"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "Timer.h"



__global__ void createIndexMapKernel(int* indicies, int* ndk, int* out_indexmap)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1];
    if (idx < n)
    {
        for (int i = 0; i < d; i++)
        {
            out_indexmap[i * n + idx] = d * indicies[idx] + i;
        }

    }

    __syncthreads();

}
void Kmeans_par_thrust(PointsData_SOA& data, int MAX_ITERS)
{
    

    // dane punktów
    float* dev_data;
    // dane centrów klastrów
    float* dev_clusterData;
    // parametry n, d, k
    int* dev_ndk;
    // tablica, w której zapisujemy 1, je¿eli zmieniliœmy centrum dla danego punktu
    int* dev_delta;
    // tablica indeksów dla funkcji reduce_by_key
    int* dev_idx_map;
    // tablica odleg³oœci miêdzy punktami a centrami
    float* dev_dist;
    // tablica indeksów klastrów dla punktów (do którego klastra przynale¿y dany punkt)
    int* dev_clusterIdx;


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    int* clusterCount = new int[k];
    float* old_clusters = new float[d * k];
    const int N_THREADS = calculateNumThreads(k);
    int delta = 0;
    int iter = 0;

    // obliczanie czasu dzialania poszczegolnych etapow
    float time_dist = 0;
    float time_index = 0;
    float time_update_clusters = 0;
    Timer_CPU t("poczatkowa inicjalizacja danych CPU -> GPU");
    // klucze dla sum wspó³rzêdnych dla nowych centrów (w zasadzie niepotrzebne)
    thrust::device_vector<float> centroidSumKeys(d * k);
    // sumy wspó³rzêdnych dla nowych centrów
    thrust::device_vector<float> centroidSums(d * k);
    // wspó³rzêdne punktów
    thrust::device_vector<float> data_thrust(d * n);
    // indeksy dla punktów pasuj¹ce do funkcji reduce_by_key
    thrust::device_vector<int> idxmap_thrust(d * n);

    // set device
    CALL(cudaSetDevice(0));

    // malloc
    CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_idx_map, n * d * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_dist, n * k * sizeof(float)));


    // memset/memcpy
    CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));
    
    std::cout << "poczatkowa inicjalizacja danych CPU -> GPU: " << t.getElapsed() << "s.\n";


    do
    {
        iter++;
        // inicjalizacja danych
        memcpy(old_clusters, data.clusterData, d * k * sizeof(float));
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_dist, 0, n * k * sizeof(float)));
        CALL(cudaMemset(dev_idx_map, 0, n * d * sizeof(int)));
        CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));

        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((n * k + (N_THREADS - 1)) / N_THREADS);

        // obliczanie odleg³oœci
        {
            Timer_CPU t("obliczanie odleglosci");
            calcDistKernel << <calcBlocks, threads, k* d * sizeof(float) >> > (dev_data, dev_clusterData, dev_ndk, dev_dist);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());
            time_dist += t.getElapsed();
        }

        dim3 updateBlocks = dim3((n + (N_THREADS - 1)) / N_THREADS);

        // obliczanie nowych indeksów klastrów dla punktów
        {
            Timer_CPU t("obliczanie nowych indeksow klastrow dla punktow");
            updateIndexKernel << <updateBlocks, threads, k* N_THREADS * sizeof(int) >> > (dev_dist, dev_ndk, dev_clusterIdx, dev_delta);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());
            time_index = t.getElapsed();
        }



        {
            Timer_CPU t("aktualizacja centrow klastrow");

            CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));
            delta = reduce(dev_delta, n);

            // obliczanie indeksów dla wspó³rzêdnych punktów
            createIndexMapKernel << <updateBlocks, threads >> > (dev_clusterIdx, dev_ndk, dev_idx_map);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());

            // kopiowanie danych do kontenerów thrusta
            Timer_GPU tgpu("a");
            thrust::copy(dev_data, dev_data + (d * n), data_thrust.begin());
            thrust::copy(dev_idx_map, dev_idx_map + (d * n), idxmap_thrust.begin());

            // obliczanie sum wspó³rzêdnych nowych centrów
            thrust::sort_by_key(idxmap_thrust.begin(), idxmap_thrust.end(), data_thrust.begin());
            thrust::reduce_by_key(idxmap_thrust.begin(), idxmap_thrust.end(), data_thrust.begin(), centroidSumKeys.begin(), centroidSums.begin());
            time_update_clusters += tgpu.getElapsed();
            CALL(cudaDeviceSynchronize());


            memset(data.clusterData, 0, sizeof(float) * k * d);
            memset(clusterCount, 0, sizeof(int) * k);

            count(dev_clusterIdx, clusterCount, n, k);


            // aktualizacja centrów klastrów
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    if (clusterCount[i] > 0)
                        data.clusterData[i * d + j] = centroidSums[i * d + j] / clusterCount[i];
                    else data.clusterData[i * d + j] = old_clusters[i * d + j];
                }
            }
            time_update_clusters = t.getElapsed();

        }

    } while (delta > 0 && iter < MAX_ITERS);


    std::cout << "Ilosc iteracji: " << iter << ".\n";
    std::cout << "delta = " << delta << ".\n";
    std::cout << "Srednie obliczanie odleglosci od punktow do centrow klastrow: " << time_dist / iter << "s.\n";
    std::cout << "Srednie obliczanie nowych indeksow centrow dla punktow: " << time_index / iter << "s.\n";
    std::cout << "Srednie obliczanie nowych centrow klastrow: " << time_update_clusters / iter << "s.\n";


    cudaFree(dev_clusterData);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_dist);
    cudaFree(dev_idx_map);

    delete old_clusters;
    delete clusterCount;
}