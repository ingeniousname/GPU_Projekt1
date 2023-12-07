#include "points_data.h"

PointsData_SOA ConvertToSOAAndFree(PointsData& data)
{
	PointsData_SOA res;
	res.ndk[0] = data.n;
	res.ndk[1] = data.d;
	res.ndk[2] = data.k;
	res.clusterData = data.clusterData;
	res.clusterIndex = data.clusterIndex;
	res.data = new float[data.n * data.d];

	for (int i = 0; i < data.n; i++)
		for (int j = 0; j < data.d; j++)
			res.data[j * data.n + i] = data.data[i * data.d + j];

	delete[] data.data;
	return res;
}

void free_data(PointsData& data)
{
	delete[] data.data;
	if (data.clusterData != nullptr)
	{
		delete[] data.clusterData;
		data.clusterData = nullptr;
	}
	if (data.clusterIndex != nullptr)
	{
		delete[] data.clusterIndex;
		data.clusterIndex = nullptr;
	}
	data.afterCalculation = false;
}

void free_data_SOA(PointsData_SOA& data)
{
	delete[] data.data;
	if (data.clusterData != nullptr)
	{
		delete[] data.clusterData;
		data.clusterData = nullptr;
	}
	if (data.clusterIndex != nullptr)
	{
		delete[] data.clusterIndex;
		data.clusterIndex = nullptr;
	}
	data.afterCalculation = false;
}
