#pragma once
#include "points_data.h"

// funkcja sprawdza, czy plik jest plikem binarnym czy nie (typowo w plikach ASCII występują tylko chary z wartościami < 127)
bool checkASCII(const char* filename);
// wrapper na funkcje czytające plik ze sprawdzaniem typu pliku
void read_data_universal(const char* filename, PointsData& data);
// czytanie danych z pliku ASCII
void read_data_ASCII(const char* filename, PointsData& data);
// czytanie danych z pliku binarnego
void read_data_binary(const char* filename, PointsData& data);