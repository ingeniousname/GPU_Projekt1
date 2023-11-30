#include "Timer.h"
#include <iostream>

Timer::Timer(std::string _message)
{
	this->message = _message;
	this->t = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
	auto now = std::chrono::high_resolution_clock::now();
	double sec = (now - t).count() / 1e9;
	std::cout << message << ": " << sec << "s\n";
}
