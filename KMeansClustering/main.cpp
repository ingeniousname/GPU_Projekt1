#include "cudaCalculation.h"
#include "data_reader.h"
#include "clustering_seq.h"
#include <iostream>
#include "Timer.h"
#include "clustering_par_thrust.cuh"
#include "clustering_par_reduce.cuh"
#include "clustering_par_common.cuh"
#include "output_writer.h"

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		printf("U¯YCIE: ./KMeansClustering <plik_wejsciowy> <plik_wyjsciowy> <iloœæ_klastrów> <numer trybu>\nTryby dzialania:\n0 - CPU\n1 - GPU (liczenie srodkow na CPU)\n2 - GPU (thrust)\n3 - GPU (zmodyfikowana funkcja redukcji)");
		return 1;
	}

	PointsData data;
	const int MAX_ITERS = 500;
	const int NUM_CLUSTERS = atoi(argv[3]);
	const int mode = atoi(argv[4]);

	{
		Timer_CPU t("Czytanie danych", true);
		try {
			read_data_universal(argv[1], data);
			initForCalculations(data, NUM_CLUSTERS);
		}
		catch (std::exception e)
		{
			std::cerr << e.what() << "\n";
			exit(1);
		}

	}
	PointsData_SOA data_SOA;

	switch (mode)
	{
	case 0:
		{
			{
				Timer_CPU t("Caly algorytm", true);
				KMeansClustering_seq(data, MAX_ITERS);
			}
		}
		break;
	case 1:
		{
			{
				Timer_CPU t("Zamiana danych na format SOA", true);
				data_SOA = ConvertToSOAAndFree(data);
			}
			{
				Timer_CPU t("Caly algorytm", true);
				Kmeans_par_UpdateClusterOnCPU(data_SOA, MAX_ITERS);
			}
		}
		break;
	case 2:
		{
			
			{
				Timer_CPU t("Zamiana danych na format SOA", true);
				data_SOA = ConvertToSOAAndFree(data);
			}
			{
				Timer_CPU t("Caly algorytm", true);
				Kmeans_par_thrust(data_SOA, MAX_ITERS);
			}
		}
		break;
	case 3:
		{
			{
				Timer_CPU t("Zamiana danych na format SOA", true);
				data_SOA = ConvertToSOAAndFree(data);
			}
			{
				Timer_CPU t("Caly algorytm", true);
				Kmeans_par_modifiedReduce(data_SOA, MAX_ITERS);
			}
		}
		break;
	default:
		std::cout << "ERROR! Nieznany tryb dzia³ania!\n";
		exit(1);
	}
	
	{
		Timer_CPU t("Wypisywanie odpowiedzi", true);
		try {

			if (mode == 0)
			{
				write_output(argv[2], data.clusterData, data.clusterIndex, data.n, data.d, data.k);
				free_data(data);
			}
			else
			{
				write_output(argv[2], data_SOA.clusterData, data_SOA.clusterIndex, data_SOA.ndk[0], data_SOA.ndk[1], data_SOA.ndk[2]);
				free_data_SOA(data_SOA);
			}
		}
		catch (std::exception e)
		{
			std::cerr << e.what() << "\n";
			exit(1);
		}
	}

	cudaDeviceReset();
	return 0;
}