#pragma once

#include <thrust/version.h>
#include <thrust/device_vector.h>

struct PointsData_Thrust
{
	thrust::device_vector<float> data;

};