#pragma once
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// klasa wg wzoru RAII licz¹ca czas na CPU
class Timer_CPU
{
	std::string message;
	bool write;
	std::chrono::high_resolution_clock::time_point t;
public:
	Timer_CPU(std::string _message, float writeout = false);
	float getElapsed();
	~Timer_CPU();

};

// klasa wg wzoru RAII licz¹ca czas na GPU
class Timer_GPU
{
	std::string message;
	bool write;
	cudaEvent_t start, stop;
public:
	Timer_GPU(std::string message, bool write = false);
	float getElapsed();
	~Timer_GPU();
};

