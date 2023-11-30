#include "clustering_seq.h"
#include "cstring"
#include "climits"
#include <iostream>

void KMeansClustering_seq(PointsData& data, int k)
{
	const float threshold = 1e-8;
	int delta = 0;
	int* clusterCount = new int[k];


	data.clusterIndex = new int[data.n];
	data.clusterData = new float[data.d * k];
	memset(data.clusterIndex, 0, data.n * sizeof(int));
	for (int i = 0; i < data.d * k; i++)
		data.clusterData[i] = data.data[i];


	do
	{
		delta = 0;
		for (int i = 0; i < data.n; i++)
		{
			int lastIdx = data.clusterIndex[i];
			int nextIdx = -1;
			float minDist = 1e10;
			for (int j = 0; j < k; j++)
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

		memset(data.clusterData, 0, data.d * k * sizeof(float));
		memset(clusterCount, 0, k * sizeof(int));
		for (int i = 0; i < data.n; i++)
		{
			clusterCount[data.clusterIndex[i]]++;
			for (int j = 0; j < data.d; j++)
			{
				data.clusterData[data.clusterIndex[i] * data.d + j] += data.data[i * data.d + j];
			}
		}

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < data.d; j++)
			{
				data.clusterData[i * data.d + j] /= clusterCount[i];
				//std::cout << data.clusterData[i * data.d + j] << ", ";
			}
		}
		//std::cout << "\n";

	} while ((float)delta / data.n >= threshold);
	data.k = k;
	data.afterCalculation = true;
	delete[] clusterCount;
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
