#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

float *a, *b;  // host data
float *c, *c2;  // results

__global__ void vecAdd(float *A,float *B,float *C,int N)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   // Limit thread execution more than its limit incase it will stop buffer overflow.  
   if(i<N){
      C[i] = A[i] + B[i];
   }
   
}

void vecAdd_h(float *A1,float *B1, float *C1, float N)
{
   for(int i=0;i<N;i++)
      C1[i] = A1[i] + B1[i];
}

int main(int argc,char **argv)
{
   printf("Begin \n");
   int n=10000000;
   int nBytes = n*sizeof(float);
   int block_size, block_no;
   a = (float *)malloc(nBytes);
   b = (float *)malloc(nBytes);
   c = (float *)malloc(nBytes);
   c2 = (float *)malloc(nBytes);
   float *a_d,*b_d,*c_d;
   block_size=1000;
   block_no = n/block_size;
   dim3 threadPerBlock(block_size,1,1);
   dim3 dimBlock(block_no,1,1);

   for(int i = 0; i < n; i++ ) {
      a[i] = sin(i)*sin(i);
      b[i] = cos(i)*cos(i);
   }
   printf("Allocating device memory on host..\n");
   cudaMalloc((void **)&a_d,nBytes);
   cudaMalloc((void **)&b_d,nBytes);
   cudaMalloc((void **)&c_d,nBytes);
   printf("Copying to device..\n");

   cudaMemcpy(a_d,a,nBytes,cudaMemcpyHostToDevice);
   cudaMemcpy(b_d,b,nBytes,cudaMemcpyHostToDevice);
   
   clock_t start_d=clock();
   printf("Doing GPU Vector add\n");
   vecAdd<<<dimBlock,threadPerBlock>>>(a_d,b_d,c_d,n);
   cudaThreadSynchronize();
   clock_t end_d = clock();
   clock_t start_h = clock();
   printf("Doing CPU Vector add\n");
   vecAdd_h(a,b,c2,n);
   clock_t end_h = clock();
   double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
   double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;
   int cpy_C=cudaMemcpy(c,c_d,nBytes,cudaMemcpyDeviceToHost);
   printf("%d\n",cpy_C);
   printf("Number of elements: %d GPU Time: %f CPU Time: %f\n",n,time_d,time_h);

   for (int i=0; i<3;i++){
      printf("%f\n", c[i]);
   }
   for (int i=0; i<3;i++){
      printf("%f\n", c2[i]);
   }
   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);   
 
   return 0;
}

