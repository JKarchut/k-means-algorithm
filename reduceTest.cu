#include <iostream>

#define N 10000

int upperbound(int x, int y)
{
    if(x % y == 0)
        return x / y;
    else
        return x / y + 1;
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

int main()
{
    int *table = new int[N];
    for(int i = 0; i < N; i++)
    {
        table[i] = 1;
    }
    int *table_d;
    cudaMalloc(&table_d, sizeof(int) * N);
    cudaMemcpy(table_d, table, sizeof(int) * N, cudaMemcpyHostToDevice);

    int thread_count = 1024;
    for(int i = N; i > 1; i =  upperbound(i, 2 * thread_count))
    {
        int blocks = upperbound(i, (thread_count * 2));
        findDelta<<<blocks,thread_count, thread_count * sizeof(int)>>>(table_d, i);
    }
    int ans;
    cudaMemcpy(&ans, table_d, sizeof(int), cudaMemcpyDeviceToHost);
    if(ans != N)
        std::cout<< ans << "oh no \n";
    return 0;
}