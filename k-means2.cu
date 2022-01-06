#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <fstream>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

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
    if (tid == 0){
        change[blockIdx.x] = sdata[0];
    } 
}

__global__ void updateCenters(
    float *objects, 
    int *membership_ordered,
    int *objects_ordered,  
    int *centers_size,
    float *centers, 
    int numObjects, 
    int numCoords,
    int numCenters)
{
    extern __shared__ int data[];
    float *sdata = (float*)data;
    int *memb_shared = (int*)&sdata[blockDim.x * numCoords]; 
    unsigned int tid = threadIdx.x;
    unsigned int dim = threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= numObjects) 
    {
        return;
    }
    sdata[tid * numCoords + dim] = objects[objects_ordered[i] * numCoords + dim];
    if(dim == 0)
    {
        memb_shared[tid] = membership_ordered[i];
    }
    __syncthreads();
    if(tid == 0)
    {
        float sumTemp = 0;
        int sizeTemp = 0;
        int cur_memb = memb_shared[0];
        for(int x = 0; x < blockDim.x && i + x < numObjects; x++)
        {
            if(cur_memb != memb_shared[x])
            {
                atomicAdd(&centers[cur_memb * numCoords + dim], sumTemp);
                if(dim == 0)
			        atomicAdd(&centers_size[cur_memb], sizeTemp);
                sumTemp = 0;
                sizeTemp = 0;
                cur_memb = memb_shared[x];
            }
            sumTemp += sdata[x * numCoords + dim];
            sizeTemp++;
        }
    } 
}

__global__ void divideCenters(float *center, int *centerSize, float *old_center, int numCoords)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(centerSize[i] != 0)
    {
        for(int x = 0; x < numCoords; x++)
        {
            center[i * numCoords + x] /= centerSize[i];
            old_center[i * numCoords + x] = center[i * numCoords + x];
        }
    }
    __syncthreads();
    if(i == 0)
    {
       for(int x = 0; x < 5; x++)
        {
            for(int y = 0; y < numCoords; y++)
            {
                printf("%f ", old_center[x * numCoords + y]);
            }
            printf("\n");
        } 
    }
    centerSize[i] = 0;
}

double GetElapsed(struct timeval begin, struct timeval end)
{
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    return (seconds + microseconds*1e-6);
}

int main(int argc, char **argv) {
    extern char   *optarg;
    int     numClusters, numCoords, numObjs, opt;
    char   *filename;
    float   *objects_h, *clusters_h;
    float *objects_d, *clusters_d;
    int *membership_d, *change_d;
    float   threshold;
    struct timeval begin, end;
    
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
    if (filename == NULL || numClusters <= 1 || numCoords < 1 || numObjs < 1 || numClusters >= numObjs) usage(argv[0], threshold);

    printf("reading data points from file %s\n",filename);

    /* allocate and set host */
    objects_h = new float[numObjs * numCoords];
    clusters_h = new float[numClusters * numCoords];

    /* read data points from file ------------------------------------------*/
    double io_timing;
    gettimeofday(&begin, 0);
    file_read(filename, objects_h, numObjs, numCoords);
    gettimeofday(&end, 0);
    io_timing = GetElapsed(begin,end);
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
    gpuErrchk(cudaMemset(clusterSize_d, 0, sizeof(int) * numClusters));
    float delta;
    int temp_delta;
    float *temp_d;
    gpuErrchk(cudaMalloc(&temp_d, sizeof(float) * numClusters * numCoords));
    gpuErrchk(cudaMemcpy(clusters_d, clusters_h, numClusters * numCoords * sizeof(float), cudaMemcpyHostToDevice));
    
    int thread_count_reduce = 1024;
    int block_count_reduce = upperbound(numObjs, thread_count_reduce);
    dim3 thread_count_centers = dim3(1024/numCoords , numCoords);
    int block_count_centers = upperbound(numObjs ,thread_count_centers.x);
    int sharedMemSize = (sizeof(int) * thread_count_centers.x) + (sizeof(float) * thread_count_centers.x * numCoords);
    thrust::device_vector<int> objects_ordered(numObjs);
    thrust::device_vector<int> membership_ordered(numObjs);
    gettimeofday(&begin, 0);
    int count = 0;
    do{
        delta = 0.0;

        //calculate closest centers
        findClosest<<<block_count_reduce,thread_count_reduce>>>(objects_d, clusters_d, membership_d, change_d, numObjs, numClusters, numCoords);
        gpuErrchk(cudaPeekAtLastError());
        
        //find delta
        for(int i = numObjs; i > 1; i =  upperbound(i, 2 * thread_count_reduce))
        {
            int blocks = upperbound(i, thread_count_reduce * 2);
            findDelta<<<blocks,thread_count_reduce, thread_count_reduce * sizeof(int)>>>(change_d, i);
            gpuErrchk(cudaPeekAtLastError());
        }
        gpuErrchk(cudaMemcpy(&temp_delta, change_d, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemset(temp_d, 0, sizeof(float) * numCoords * numClusters));

        thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(numObjs),
                 objects_ordered.begin());
        thrust::copy(membership_d, &membership_d[numObjs - 1], membership_ordered.begin());
        thrust::sort_by_key(membership_ordered.begin(),
                        membership_ordered.end(),
                        objects_ordered.begin());
        // calculate new centers sum and centerSize

        updateCenters<<<block_count_centers, thread_count_centers, sharedMemSize>>>
        (objects_d, thrust::raw_pointer_cast(membership_ordered.data()), thrust::raw_pointer_cast(objects_ordered.data()), clusterSize_d, temp_d, numObjs, numCoords, numClusters);
        gpuErrchk( cudaPeekAtLastError());
        
        // divide centers sum by size
        divideCenters<<<1, numClusters>>>(temp_d, clusterSize_d, clusters_d, numCoords);
        gpuErrchk( cudaPeekAtLastError());
        
        delta = temp_delta;
        delta /= numObjs;
        printf("%f\n",delta);
        count++;
    }while(delta > threshold && count < 100);
    gettimeofday(&end, 0);
    double clustering_timing = GetElapsed(begin,end);

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
    free(clusters_h);

    cudaFree(objects_d);
    cudaFree(change_d);
    cudaFree(membership_d);
    cudaFree(clusters_d);
    cudaFree(temp_d);
    cudaFree(clusterSize_d);

    printf("\nPerforming **** Regular Kmeans (parallel version 1 (centers using reduce)) ****\n");
    printf("Input file:     %s\n", filename);
    printf("numObjs       = %d\n", numObjs);
    printf("numCoords     = %d\n", numCoords);
    printf("numClusters   = %d\n", numClusters);
    printf("threshold     = %.4f\n", threshold);

    printf("I/O time           = %10.4f sec\n", io_timing);
    printf("Computation timing = %10.4f sec\n", clustering_timing);
    return(0);
}