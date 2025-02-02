<h2>A GPU-accelerated k-means clustering method implementation</h2>

This program is an implementation of the k-means clustering method using C++/CUDA.
 
As an input, program takes a file with number of dimensions $d$, number of data points $n$, and data points as $d$-dimensional vectors. The generateData script helps with creating the right data format for testing as well as allows for binary encoding which speeds up the reading process. The parameter $k$ - number of clusters to find - is also taken as an input.

There are 4 approaches to choose from in the program:
- Fully CPU-oriented solution
- Calculate vector-centroid distances on GPU, find new clusters on CPU
- Fully GPU-oriented solution (calculating centroids using Thrust library)
- Fully GPU-oriented solution (calculating centroids using custom reduction kernels)

As an input parameter the program takes an index of the method which allows for easy comparisons.

As an output the program writes out to an output file $k$ cluster vectors and then $n$ indexes - the $i$-th index describes to which of the cluster vectors the $i$-th vector from the input file belongs to.