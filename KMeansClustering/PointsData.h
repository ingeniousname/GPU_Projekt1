#pragma once

struct PointsData
{
	bool afterCalculation = false;
	float* data = nullptr;
	float* clusterData = nullptr;
	int* clusterIndex = nullptr;
	int k;
	int n;
	int d;
};

struct PointsData_FlatArray
{
	bool afterCalculation = false;
	int ndk[3];
	int* clusterIndex = nullptr;
	float* clusterData = nullptr;
	float* data = nullptr;
};


PointsData_FlatArray ConvertToSeqAndFree(PointsData& data);