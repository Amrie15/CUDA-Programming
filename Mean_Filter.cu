#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

const int N=640;
const int window=5;

__global__ void mean_Filter (int *inputImage, int *outputImage , int window, int N) {
    window=window/2;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * N;
 
    outputImage[index] = inputImage[index]; 
    //  Neglect matrix edge value
    if(col < N-window && row < N-window && row >= window && col >= window) {
        int sum = 0;
        for(int x = -window; x<=window ; x++) {
            for(int y = -window; y<= window; y++) {
                    // N*x control row, y control column 
                    sum += inputImage[index + N*x + y]; 
            }
        }
        outputImage[index] = sum/((window*2+1)*(window*2+1));
    }
 }
/**
void mean_Filter_h(int *inputImage, int *outputImage,int window, int N){
    window=window/2;
    int sum=0;
    for (int row=window;row<(rows-window); row++){
        for(int col =window; col<(cols-window);col++){
            sum=0;
            for(int i=-window;i<=window;i++){
                for(int j=-window;j<=window;j++){
                    sum=sum+inputImage[row+i][col+j];
                }
            }
            output[row][col]=
        }
    }
}
*/


int main(int argc, char**argv)
{
    // Input image and output image for the testing
    int inputImgageS[N][N], outputImgageS[N][N];
    // Input and output image for the filering 
    int *inputImage, *outputImage;

    // Number of elements in a 2D array image
    int size = N * N * sizeof(int);

    // Create a image5(2D Array) with random numbers
    for (int row=0; row<N; row++) {
        for (int col=0; col<N; col++){
            inputImgageS[row][col] = rand() % 256;
        }
    }

    // Allocate memory on the GPUs for the image
    cudaMalloc((void**)&inputImage, size);
    cudaMalloc((void**)&outputImage, size);
    // Copy the image form host to device (input and output)
    cudaMemcpy(inputImage, &inputImgageS, size, cudaMemcpyHostToDevice);
    cudaMemcpy(outputImage, &inputImgageS, size, cudaMemcpyHostToDevice);

    //  Total Number of threads in a block 32*32 = 1024 <= 1024(Threads per block)
    dim3 threadsPerBlock(32,32); 
    //  For 640*640 , 20*20 block For 1280*1280 , 40*40
    dim3 blocksForGrid(N/threadsPerBlock.x, N/threadsPerBlock.y);  
    
    // GPUs mean filter, time start
    clock_t start_d=clock();
    printf("Doing GPU mean filter\n");
    // GPUs' mean filtering 
    mean_Filter<<<blocksForGrid, threadsPerBlock>>>(inputImage,outputImage,window,N);
    cudaDeviceSynchronize();
    clock_t end_d = clock();

    // CPUs mean filter, time start
    // clock_t start_h = clock();
    // printf("Doing CPU mean filter\n");
    // mean_Filter_h(inputImgageS,outputImgageS,window,N);
    // clock_t end_h = clock();

    //  Time calculation
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    // double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    // Copy GPUs' output image to Host 
    cudaMemcpy(&outputImgageS, outputImage , size, cudaMemcpyDeviceToHost);

    // printf("Image size : %d Window size : %d GPU Time: %f CPU Time: %f\n",N,window,time_d,time_h);
    printf("Image size : %d Window size : %d GPU Time: %f \n",N,window,time_d);
    
    // Print Imput image
    printf("Input image \n");
    for (int row=0;row< N;row++){
        printf("[");
        for(int col=0;col<N;col++){
            printf("%d,",inputImgageS[row][col]);
        }
        printf("]\n");
    }

    // Print output image 
    printf("Output Image\n");
    for (int row=0;row< N;row++){
        printf("[");
        for(int col=0;col<N;col++){
            printf("%d,",outputImgageS[row][col]);
        }
        printf("]\n");
    }


    // Free memory on GPU 
    cudaFree(inputImage);
    cudaFree(outputImage);

    return 0;
    
}