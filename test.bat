XCOPY .\x64\Release\KMeansClustering.exe .\KMeansClustering\
CD  .\KMeansClustering
compute-sanitizer --tool memcheck ./KMeansClustering.exe C:\Users\milos\source\repos\KMeansClustering\KMeansClustering\data\data ./out 32 1
PAUSE
CD .\..\