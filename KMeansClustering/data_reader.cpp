#include "data_reader.h"
#include <fstream>



void read_data(const char* filename, PointsData& data)
{
    std::ifstream f(filename, std::ios::in);
    f >> data.d;
    f >> data.n;
    data.data = new float[data.n * data.d];
    for (int i = 0; i < data.n * data.d; i++)
        f >> data.data[i];
    f.close();
}

void read_data_binary(const char* filename, PointsData& data)
{
    std::ifstream f(filename, std::fstream::binary);
    f.read((char*)&data.d, sizeof(int));
    f.read((char*)&data.n, sizeof(int));
    data.data = new float[data.n * data.d];
    f.read((char*)data.data, sizeof(float) * data.d * data.n);
    f.close();
}

void free_data(PointsData& data)
{
    delete[] data.data;
    if (data.clusterData != nullptr)
        delete[] data.clusterData;
    if (data.clusterIndex != nullptr)
        delete[] data.clusterIndex;
    data.afterCalculation = false;
}
