#include "cudaCalculation.h"
#include "data_reader.h"
#include "clustering_seq.h"
#include <iostream>
#include "Timer.h"
#include "clustering_par.h"
#include "update_index_naive.h"

int main()
{
	PointsData data1, data2;
	const int NUM_CLUSTERS = 2;
	PointsData_FlatArray data_flat;
	{
		Timer t("Data reading");
		read_data_binary("./data/binary_data1m", data1);
		read_data_binary("./data/binary_data1m", data2);
		initForCalculations(data2, NUM_CLUSTERS);
		//initForCalculations(data1, NUM_CLUSTERS);
		data_flat = ConvertToSeqAndFree(data2);
	}
	//for (int i = 0; i < data.n; i++)
	//{
	//	for (int j = 0; j < data.d; j++)
	//		std::cout << data.data[i * data.d + j] << " ";
	//	std::cout << "\n";
	//}
	//KMeansClustering_seq(data, 2);
	//initForCalculations(data1, 2);
	//float* dist = new float[data.n * data.k];
	//calcDistSeq(data, dist);
	{
		Timer t("seq");
		KMeansClustering_seq(data1, NUM_CLUSTERS);
		
	
	}
	{
		Timer t("par");
		Kmeans_par_NewClusterSumGPU(data_flat, NUM_CLUSTERS);
		//Kmeans_par_NewClusterSumGPUWithThrust(data_flat, NUM_CLUSTERS);
	}
	//for (int i = 0; i < data.n * data.d; i++)
	//	std::cout << dist[i] << "\n";
	//int checksum = 0;
	//for (int i = 0; i < data1.n; i++)
	//	checksum += data1.clusterIndex[i] - data2.clusterIndex[i];
	//std::cout << checksum << "\n";
	//delete[] dist;
	
	for (int i = 0; i < data_flat.ndk[2]; i++)
	{
		std::cout << "C" << i << ": ";
		for (int j = 0; j < data_flat.ndk[1]; j++)
		{
			std::cout << data_flat.clusterData[i * data_flat.ndk[1] + j] << " ";
		}
		std::cout << "\n";
	}
	cudaDeviceReset();
	//free_data(data);
	return 0;
}