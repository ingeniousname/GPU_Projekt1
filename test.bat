XCOPY .\x64\Debug\KMeansClustering.exe .\KMeansClustering\
CD  .\KMeansClustering
compute-sanitizer --tool memcheck ./KMeansClustering.exe
PAUSE
CD .\..\