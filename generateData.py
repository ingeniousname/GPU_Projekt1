import sys, getopt, random
import numpy as np
import struct

helpstring = """Skrypt służący do generowania n-elementowego zbioru d-wymiarowych punktów, z czego punkty te zbierają się w k różnych klastrów.
do każdego z klastrów należy ok. n/k punktów, program losuje d*k liczb z przedziału <0, 1>, a następnie losuje punkty korzystając z wylosowanych wartości jako wartość średnia w rozkładzie normalnym, z odchyleniem standardowym 0.1, Dostępne opcje:
-n<liczba> - n = liczba
-d<liczba> - d = liczba
-k<liczba> - k = liczba
-b - zapis binarny, zalecany dla dużych danych (bez tej opcji zapis tekstowy)
-o - ścieżka do plik wyjściowego
-s - wygeneruj plik z odpowiedzą do pliku ./solution.txt (zawierający środki klastrów oraz odpowiednie przyporządkowanie)
Przykładowe wywołanie:
python3 ./generateData.py -b -s -n1000000 -k7 -d7 -o ./data_n1md7"""



d = 2
k = 2
n = 100
binary = False
solution = False
out = "data"
opts, args = getopt.getopt(sys.argv[1:],"hbsd:n:o:k:")
for opt, arg in opts:
   if opt == '-b':
      binary = True
   if opt == '-s':
      solution = True
   if opt == '-h':
      print (helpstring)
      sys.exit()
   elif opt == "-k":
      k = int(arg)
   elif opt == "-d":
      d = int(arg)
   elif opt == "-n":
      n = int(arg)
   elif opt == "-o":
      out = arg

div = int(n/k)
ans = []

means = []
for i in range(k):
   tmp = []
   for i in range(d):
      tmp.append(random.random())
   means.append(tmp)
print(means)

if(binary):
   binary_data = bytearray()
   binary_data.extend(struct.pack('i', d))
   binary_data.extend(struct.pack('i', n))
   for i in range(n):
      t = random.randint(0, k - 1)
      ans.append(t)
      for j in range(d):
         binary_data.extend(struct.pack('f', float(np.random.normal(means[t][j], 0.1))))
   with open(out, "wb") as f:
      f.write(binary_data)
else:
   with open(out, "w+") as f:
      f.write(f'{d} {n}\n')
      f.write("\n")
      for i in range(n):
         t = random.randint(0, k - 1)
         ans.append(t)
         for j in range(d):
           f.write(f'{str(np.random.normal(means[t][j], 0.1))} ')          
         f.write("\n")

if(solution):
   with open("./solution.txt", "w") as f:
      for i in range(0, k):
         f.write(f'C{i}: ')
         for j in range(0, d):
            f.write("{:.4f} ".format(means[i][j]))
         f.write(f'\n')
         
      f.write(f'\n')
      for i in range(0, n):
         f.write(f'{i} {ans[i]}\n')
          
