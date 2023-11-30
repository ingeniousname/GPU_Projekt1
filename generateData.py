import sys, getopt, random
import numpy as np
import struct

d = 2
k = 2
n = 100
binary = False
out = "data.txt"
opts, args = getopt.getopt(sys.argv[1:],"hbd:n:o:k:")
for opt, arg in opts:
   if opt == '-b':
      binary = True
   if opt == '-h':
      print ('test.py -d <dimensions> -n <num_points> -o <output_file>')
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
      t = int(i/n*k)
      for j in range(d):
         binary_data.extend(struct.pack('f', float(np.random.normal(means[t][j], 0.1))))
   with open(out, "wb") as f:
      f.write(binary_data)
else:
   with open(out, "w+") as f:
      f.write(f'{d} {n}\n')
      f.write("\n")
      for i in range(n):
         t = int(i/div)
         for j in range(d):
           f.write(f'{str(np.random.normal(means[t][j], 0.1))} ')          
         f.write("\n")
          
