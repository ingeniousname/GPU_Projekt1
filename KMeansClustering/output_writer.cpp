#include "output_writer.h"
#include <fstream>
#include <iostream>

void write_output(const char* filename, float* clusterData, int* indicies, int n, int d, int k)
{
	std::ofstream out_file(filename, std::ios::out);
	if (!out_file.is_open())
		throw std::runtime_error("Failed to open output file!");
	for (int i = 0; i < k; i++)
	{
		out_file << "C" << i << ": ";
		for (int j = 0; j < d; j++)
		{
			out_file << clusterData[i * d + j] << " ";
		}
		out_file << "\n";
	}
	out_file << "\n";
	for (int i = 0; i < n; i++)
		out_file << i << " " << indicies[i] << "\n";


	out_file.close();
}
