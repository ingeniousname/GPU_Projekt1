#include "Timer.h"
#include <iostream>

Timer_CPU::Timer_CPU(std::string _message, float writeout)
{
	this->write = writeout;
	this->message = _message;
	this->t = std::chrono::high_resolution_clock::now();
}

float Timer_CPU::getElapsed()
{
	auto now = std::chrono::high_resolution_clock::now();
	return (now - t).count() / 1e9;
}

Timer_CPU::~Timer_CPU()
{
	if (write)
	{
		auto now = std::chrono::high_resolution_clock::now();
		double sec = (now - t).count() / 1e9;
		std::cout << message << ": " << sec << "s\n";
	}
}

Timer_GPU::Timer_GPU(std::string message, bool write)
{
	this->write = write;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

float Timer_GPU::getElapsed()
{
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	return time / 1e3;
}

Timer_GPU::~Timer_GPU()
{
	if (write)
	{
		cudaEventRecord(stop, 0);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		std::cout << message << ": " << time << "s\n";
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}
