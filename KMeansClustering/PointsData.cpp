#include "PointsData.h"

PointsData_FlatArray ConvertToSeqAndFree(PointsData& data)
{
	PointsData_FlatArray res;
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
