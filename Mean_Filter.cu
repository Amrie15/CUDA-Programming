#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

// host input
int *a, *b;
// Host output
int *c;

__global__ void vecAdd(int *A, int *B, int*C){
    int index=threadIdx.x + blockIdx.x * blockDim.x;

    C[index]=A[index]+B[index];

}



int main(int argc, char**argv)
{
    int n = 20;
    
    

    int nBytes= n*sizeof(int);
    a=(int *)malloc(nBytes);
    b=(int *)malloc(nBytes);
    c=(int *)malloc(nBytes);

    for (int i=0; i<20;i++){
        a[i]=i+1;
        b[i]=i+1;
    }
    for (int i=0; i<20;i++){
        printf("%d \n", a[i]);
    }
    
    int *a_d, *b_d, *c_d;

    cudaMalloc((void**)&a_d, nBytes);
    cudaMalloc((void**)&b_d, nBytes);
    cudaMalloc((void**)&c_d, nBytes);

    
    cudaMemcpy(a_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(b_d,b,n*sizeof(int),cudaMemcpyHostToDevice);

    vecAdd<<<1, 20>>>(a_d, b_d, c_d);
    cudaThreadSynchronize();

    cudaMemcpy(c, c_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice);
    

    for (int i=0;i<20;i++){
        printf("%d \n", c[i]);
    }

    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}