#pragma once
#include "points_data.h"

// funkcja sprawdza, czy plik jest plikem binarnym czy nie (typowo w plikach ASCII wystêpuj¹ tylko chary z wartoœciami < 127)
bool checkASCII(const char* filename);
// wrapper na funkcje czytaj¹ce plik ze sprawdzaniem typu pliku
void read_data_universal(const char* filename, PointsData& data);
// czytanie danych z pliku ASCII
void read_data_ASCII(const char* filename, PointsData& data);
// czytanie danych z pliku binarnego
void read_data_binary(const char* filename, PointsData& data);