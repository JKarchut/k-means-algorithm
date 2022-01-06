all : k-means1 k-means2

k-means1: k-means1.cu
	nvcc k-means1.cu -o k-means1
	
k-means2: k-means2.cu
	nvcc k-means2.cu -o k-means2