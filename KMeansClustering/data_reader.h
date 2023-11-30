#pragma once
#include "PointsData.h"

void read_data(const char* filename, PointsData& data);
void read_data_binary(const char* filename, PointsData& data);
void free_data(PointsData& data);