#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <fstream>
#include <cfloat>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -k num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -n dim_number  : number of dimensions\n"
        "       -N num_objects : number of objects\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

int upperbound(int x, int y)
{
    if(x % y == 0)
        return x / y;
    else
        return x / y + 1;
} 

void file_read(char *filename, float* objects, int numObjs, int numCoords)
{
    std::ifstream input(filename);
    int id;
    int i = 0;
    int j = 0;
    while(input >> id)
    {
        while(j < numCoords)
        {
            input>> objects[i * numCoords + j];
            j++;
        }
        j = 0;
        i++;
    }
    input.close();
}

__global__ void findClosest(
    float *objects, 
    float *centers, 
    int *membership,  
    int *change, 
    int numObj,
    int numClust, 
    int numCoord)
{
    int objId = blockIdx.x * blockDim.x + threadIdx.x;
    if(objId >= numObj) return;
    int clustId;
    int initialMembership = membership[objId];
    float minDist = FLT_MAX;
    change[objId] = 0;
    for(int i = 0; i < numClust; i++)
    {
        float dist = 0;
        for(int j = 0; j < numCoord; j++)
        {
            dist += (objects[objId * numCoord + j] - centers[i * numCoord + j]) * (objects[objId * numCoord + j] - centers[i * numCoord + j]); 
        }
        if(dist < minDist)
        {
            minDist = dist;
            clustId = i;
        }
    }
    membership[objId] = clustId;
    if(initialMembership != clustId)
        change[objId] = 1;
}

__global__ void findDelta(int* change, int numObj)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x* (blockDim.x * 2) + threadIdx.x;
    if(i >= numObj) 
    {
        sdata[tid] = 0;
        return;
    }
    sdata[tid] = change[i];
    if(i + blockDim.x < numObj)
        sdata[tid] += change[i + blockDim.x];
    __syncthreads();
    for (unsigned int s=blockDim.x / 2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) change[blockIdx.x] = sdata[0];
}

__global__ void updateCenters(
    float *objects, 
    int *membership, 
    float *centers, 
    int *center_size, 
    int numObjects, 
    int numCoords,
    int numCenters)
{
    extern __shared__ int s[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numObjects) return;
	int tid = threadIdx.x;
	float *s_data = (float*)s;
    float *centers_sum_temp = (float*)&s_data[numCoords * blockDim.x];
    for(int x = 0; x < numCoords; x++)
    {
        s_data[tid + x]= objects[i * numCoords + x];
        if(tid < numCenters)
            centers_sum_temp[tid + x] = 0;
    }
	int *center_assingment = (int*)&centers_sum_temp[numCoords * numCenters];
    int *centers_size_temp = (int*)&center_assingment[blockDim.x];
	center_assingment[tid] = membership[i];
    if(tid < numCenters)
        centers_size_temp[tid] = 0;

	__syncthreads();

	if(tid == 0)
	{
		for(int j=0; j< blockDim.x; ++j)
		{
			int clust_id = center_assingment[j];
            for(int x = 0; x < numCoords; x++)
			    centers_sum_temp[clust_id + x]+=s_data[j * numCoords + x];
			centers_size_temp[clust_id]+=1;
		}

		for(int z=0; z < numCenters; z++)
		{
            for(int x = 0; x < numCoords; x++)
			    atomicAdd(&centers[z * numCoords + x],centers_sum_temp[z * numCoords + x]);
			atomicAdd(&center_size[z],centers_size_temp[z]);
		}
	}
}

__global__ void divideCenters(float *center, int *centerSize, float *old_center, int numCenters, int numCoords)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int x = 0; x < numCoords; x++)
    {
        if(centerSize[i] != 0)
        {
            center[i * numCoords + x] /= centerSize[i];
            old_center[i * numCoords + x] = center[i * numCoords + x];
        }
        printf("%f ", old_center[i * numCoords + x]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    extern char   *optarg;
    int     numClusters, numCoords, numObjs, opt;
    char   *filename;
    float   *objects_h, *clusters_h;
    float *objects_d, *clusters_d;
    int *membership_d, *change_d, *change_h, *newClusterSize, *membership_h;
    float   threshold;

    /* some default values */
    threshold   = 0.001;
    numClusters = 0;
    filename    = NULL;
    numCoords   = 0;
    numObjs     = 0;
    
    while ( (opt=getopt(argc,argv,"i:n:t:N:k:"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': numClusters = atoi(optarg);
                      break;
            case 'N': numObjs = atoi(optarg);
                      break;
            case 'n': numCoords = atoi(optarg);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }
    std::cout << numObjs << ' ' << numCoords << ' ' << numClusters << std::endl;
    if (filename == NULL || numClusters <= 1 || numCoords < 1 || numObjs < 1 || numClusters >= numObjs) usage(argv[0], threshold);

    printf("reading data points from file %s\n",filename);

    /* allocate and set host */
    objects_h = new float[numObjs * numCoords];
    clusters_h = new float[numClusters * numCoords];

    /* read data points from file ------------------------------------------*/

    file_read(filename, objects_h, numObjs, numCoords);

    /* allocate and copy data to device */
    for(int i = 0; i < numCoords * numClusters; i++)
    {
        clusters_h[i] = objects_h[i];
    }
    int *clusterSize_d;
    gpuErrchk(cudaMalloc(&clusterSize_d, numClusters * sizeof(int)));
    gpuErrchk(cudaMalloc(&objects_d, numObjs * numCoords * sizeof(float)));
    gpuErrchk(cudaMalloc(&clusters_d, numClusters * numCoords * sizeof(float)));
    gpuErrchk(cudaMalloc(&membership_d, sizeof(int) * numObjs));
    gpuErrchk(cudaMalloc(&change_d, sizeof(int) * numObjs));
    gpuErrchk(cudaMemcpy(objects_d, objects_h, numObjs * numCoords * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(membership_d, -1, sizeof(int) * numObjs));

    int thread_count = 1024;
    int block_count = upperbound(numObjs, thread_count);

    newClusterSize = new int[numClusters];
    change_h = new int[numObjs];
    membership_h = new int[numObjs];
    float delta;
    float *temp_d;
    gpuErrchk(cudaMalloc(&temp_d, sizeof(float) * numClusters * numCoords));
    gpuErrchk(cudaMemcpy(clusters_d, clusters_h, numClusters * numCoords * sizeof(float), cudaMemcpyHostToDevice));
    do{
        delta = 0.0;
        findClosest<<<block_count,thread_count>>>(objects_d, clusters_d, membership_d, change_d, numObjs, numClusters, numCoords);
        gpuErrchk( cudaPeekAtLastError());
        
        for(int i = numObjs; i > 1; i =  upperbound(i, 2 * thread_count))
        {
            int blocks = upperbound(i, thread_count * 2);
            findDelta<<<blocks,thread_count, thread_count * sizeof(int)>>>(change_d, i);
            gpuErrchk( cudaPeekAtLastError());
        }
        cudaMemcpy(&delta, change_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(clusters_d, 0, sizeof(float) * numCoords * numClusters);
        int sharedMemSize = (sizeof(int) * thread_count) + (sizeof(float) * thread_count * numCoords) + (numClusters * sizeof(int)) + (numClusters * numCoords * sizeof(float));
        updateCenters<<<upperbound(numObjs, thread_count), thread_count, sharedMemSize>>>(objects_d, membership_d, temp_d, clusterSize_d, numObjs, numCoords, numClusters);
        gpuErrchk( cudaPeekAtLastError());
        divideCenters<<<1, numClusters>>>(temp_d, clusterSize_d, clusters_d , numClusters, numCoords);
        gpuErrchk( cudaPeekAtLastError());
        std:: cout << delta << std::endl;
        delta /= numObjs;
        std::cout << delta << std::endl;
    }while(delta > threshold);
    gpuErrchk(cudaMemcpy(clusters_h, clusters_d, sizeof(float) * numClusters * numCoords, cudaMemcpyDeviceToHost));
    std::ofstream output("output.txt");
    for(int i = 0; i < numClusters; i++)
    {
        output << i << ' ';
        for(int j = 0; j < numCoords; j++)
        {
            output << clusters_h[i * numCoords + j] << ' ';
        }
        output << '\n';
    }
    output.close();
    free(objects_h);
    free(membership_h);
    free(change_h);
    free(newClusterSize);
    free(clusters_h);
    cudaFree(objects_d);
    cudaFree(change_d);
    cudaFree(membership_d);
    cudaFree(clusters_d);

    printf("\nPerforming **** Regular Kmeans (sequential version) ****\n");
    printf("Input file:     %s\n", filename);
    printf("numObjs       = %d\n", numObjs);
    printf("numCoords     = %d\n", numCoords);
    printf("numClusters   = %d\n", numClusters);
    printf("threshold     = %.4f\n", threshold);
    return(0);
}