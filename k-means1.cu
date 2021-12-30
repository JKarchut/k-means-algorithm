#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <fstream>
#include <cfloat>

/*---< usage() >------------------------------------------------------------*/
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

__global__ void findDelta(int* change, int numObj, int* total)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x* (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = change[i] + change[i+blockDim.x];
    __syncthreads();
    for (unsigned int s=blockDim.x / 2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) total[blockIdx.x] = sdata[0];
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
    
    while ( (opt=getopt(argc,argv,"i:n:t:N:n"))!= EOF) {
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
    if (filename == 0 || numClusters <= 1 || numCoords < 1 || numObjs < 1 || numClusters < numObjs) usage(argv[0], threshold);

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
    cudaMalloc(&objects_d, numObjs * numCoords * sizeof(float));
    cudaMalloc(&clusters_d, numClusters * numCoords * sizeof(float));
    cudaMalloc(&membership_d, sizeof(int) * numObjs);
    cudaMalloc(&change_d, sizeof(int) * numObjs);
    cudaMemcpy(objects_d, objects_h, numObjs * numCoords * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(membership_d, -1, sizeof(int) * numObjs);

    /* start the timer for the core computation -----------------------------*/
    /* membership: the cluster id for each data object */ 
    int thread_count = 1024;
    int block_count = (numObjs / thread_count) + 1;

    newClusterSize = new int[numClusters];
    change_h = new int[numObjs];
    membership_h = new int[numObjs];
    float delta;
    
    do{
        delta = 0.0;
        memset(newClusterSize, 0, sizeof(int) * numClusters);
        memset(clusters_h, 0, sizeof(float) * numCoords * numClusters);
        cudaMemcpy(clusters_d, clusters_h, numClusters * numCoords * sizeof(float), cudaMemcpyHostToDevice);
        findClosest<<<block_count,thread_count>>>(objects_d, clusters_d, membership_d, change_d, numObjs, numClusters, numCoords);

        cudaMemcpy(change_h, change_d, sizeof(int) * numObjs, cudaMemcpyDeviceToHost);
        cudaMemcpy(membership_h, membership_d, sizeof(int) * numObjs, cudaMemcpyDeviceToHost);
        for(int i = 0; i < numObjs; i++)
        {
            delta += change_h[i];
            newClusterSize[membership_h[i]]++;
            for(int j = 0; j < numCoords; j++)
            {
                clusters_h[membership_h[i] * numCoords + j] += objects_h[i * numCoords + j];
            }
        }
        for(int i = 0; i < numClusters; i++)
            for(int j = 0; j < numCoords; j++)
            {
                clusters_h[i * numCoords + j] /=newClusterSize[i];
            }
        delta /= numObjs;
    }while(delta > threshold);
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