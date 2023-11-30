#pragma once
#include <string>
#include <chrono>


class Timer
{
	std::string message;
	std::chrono::high_resolution_clock::time_point t;
public:
	Timer(std::string _message);
	~Timer();

};

