#pragma once


// struktura przechowuj¹ca dane w formacie do przetwarzania na CPU
struct PointsData
{
	bool afterCalculation = false;
	float* data = nullptr;
	float* clusterData = nullptr;
	int* clusterIndex = nullptr;
	int k;
	int n;
	int d;
};

// struktura przechowuj¹ca dane w formacie do przetwarzania na GPU (zapewnia inne u³o¿enie danych)
struct PointsData_SOA
{
	bool afterCalculation = false;
	int ndk[3];
	int* clusterIndex = nullptr;
	float* clusterData = nullptr;
	float* data = nullptr;
};

// Konwersja z typu PointsData na PointsData_SOA
PointsData_SOA ConvertToSOAAndFree(PointsData& data);
// zwalnianie danych ze struktury PointsData
void free_data(PointsData& data);
// zwalnianie danych ze struktury PointsData_SOA
void free_data_SOA(PointsData_SOA& data);