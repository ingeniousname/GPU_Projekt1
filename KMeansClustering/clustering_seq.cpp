#include "clustering_seq.h"
#include "cstring"
#include "climits"
#include "Timer.h"
#include <iostream>

void KMeansClustering_seq(PointsData& data, int MAX_ITERS)
{
	int delta = 1;
	int iter = 0;
	int* clusterCount = new int[data.k];

	float time_dist_calc = 0;
	float time_cluster_calc = 0;

	while(delta > 0 && iter < MAX_ITERS)
	{
		iter++;
		delta = 0;
		{
			Timer_CPU t("obliczanie odleglosci i obliczanie nowych indeksow klastrow dla punktow");
			for (int i = 0; i < data.n; i++)
			{
				int lastIdx = data.clusterIndex[i];
				int nextIdx = -1;
				float minDist = 1e30;
				for (int j = 0; j < data.k; j++)
				{
					float d = dist(data, i, j);
					if (d < minDist)
					{
						nextIdx = j;
						minDist = d;
					}
				}

				data.clusterIndex[i] = nextIdx;
				if (lastIdx != nextIdx)
					delta++;
			}
			time_dist_calc += t.getElapsed();
		}

		memset(data.clusterData, 0, data.d * data.k * sizeof(float));
		memset(clusterCount, 0, data.k * sizeof(int));
		{
			Timer_CPU t("aktualizacja centrow klastrow");
			for (int i = 0; i < data.n; i++)
			{
				clusterCount[data.clusterIndex[i]]++;
				for (int j = 0; j < data.d; j++)
				{
					data.clusterData[data.clusterIndex[i] * data.d + j] += data.data[i * data.d + j];
				}
			}
			
			for (int i = 0; i < data.k; i++)
			{
				for (int j = 0; j < data.d; j++)
				{
					data.clusterData[i * data.d + j] /= clusterCount[i];
				}
			}
			time_cluster_calc += t.getElapsed();
		}

	}
	data.afterCalculation = true;
	delete[] clusterCount;
	std::cout << "Ilosc iteracji: " << iter << ".\n";
	std::cout << "delta = " << delta << ".\n";
	std::cout << "Sredni czas obliczania indeksow nowych centrow dla punktow: " << time_dist_calc / iter << "s.\n";
	std::cout << "Sredni czas obliczania nowych centrow: " << time_cluster_calc / iter << "s.\n";
}


void calcDistSeq(PointsData& data, float* res)
{

	for (int i = 0; i < data.n; i++)
	{
		for (int j = 0; j < data.k; j++)
		{
			res[i * data.k + j] = dist(data, i, j);
		}
	}
}

int calcNewIndicies(PointsData& data)
{
	int delta = 0;
	float* dist = new float[data.n * data.k];
	calcDistSeq(data, dist);
	for (int i = 0; i < data.n; i++)
	{
		float dist_min = dist[i * data.k];
		int idx_min = 0;
		for (int j = 1; j < data.k; j++)
		{
			if (dist_min > dist[i * data.k + j])
			{
				idx_min = j;
				dist_min = dist[i * data.k + j];
			}
		}
		if (idx_min != data.clusterIndex[i])
		{
			delta++;
			data.clusterIndex[i] = idx_min;
		}
	}
	delete[] dist;
	return delta;
}


void initForCalculations(PointsData& data, int k)
{
	data.k = k;
	data.clusterIndex = new int[data.n];
	data.clusterData = new float[data.d * k];
	memset(data.clusterIndex, 0, data.n * sizeof(int));
	for (int i = 0; i < data.d * k; i++)
		data.clusterData[i] = data.data[i];
}

float dist(PointsData& data, int dataidx, int clusteridx)
{
	float res = 0.0f;
	for (int i = 0; i < data.d; i++)
	{
		float diff = data.data[dataidx * data.d + i] - data.clusterData[clusteridx * data.d + i];
		res += diff * diff;
	}
	return res;
}
