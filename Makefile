all : k-means1

k-means1:
	nvcc k-means1.cu -o k-means1
	