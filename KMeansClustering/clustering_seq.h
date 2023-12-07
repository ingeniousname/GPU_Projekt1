#pragma once

#include "data_reader.h"

void KMeansClustering_seq(PointsData& data, int MAX_ITERS);
void calcDistSeq(PointsData& data, float* res);
int calcNewIndicies(PointsData& data);
void initForCalculations(PointsData& data, int k);
float dist(PointsData& data, int dataidx, int clusteridx);
