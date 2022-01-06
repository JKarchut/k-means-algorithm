all : k-means1 k-means2

k-means1:
	nvcc k-means1.cu -o k-means1
	
k-means2:
	nvcc k-means2.cu -o k-means2