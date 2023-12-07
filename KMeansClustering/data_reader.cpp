#include "data_reader.h"
#include <fstream>


bool checkASCII(const char* filename)
{
    bool is_ascii = true;
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open())
        throw std::runtime_error("Failed to open input file!");
    int line;
    while ((is_ascii) && (!file.eof())) {
        line = file.get();
        if (line > 127) {
            is_ascii = false;
        }
    }
    file.close();

    return is_ascii;
}

void read_data_universal(const char* filename, PointsData& data)
{
    if (checkASCII(filename))
    {
        read_data_ASCII(filename, data);
    }
    else
    {
        read_data_binary(filename, data);
    }
}

void read_data_ASCII(const char* filename, PointsData& data)
{
    std::ifstream f(filename, std::ios::in);
    if (!f.is_open())
        throw std::runtime_error("Failed to open input file!");
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
    if (!f.is_open())
        throw std::runtime_error("Failed to open input file!");
    f.read((char*)&data.d, sizeof(int));
    f.read((char*)&data.n, sizeof(int));
    data.data = new float[data.n * data.d];
    f.read((char*)data.data, sizeof(float) * data.d * data.n);
    f.close();
}
