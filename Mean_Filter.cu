#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

// const int N=1280;
// const int window=3;

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

void mean_Filter_h(int *inputImage, int *outputImage,int window, int N){
    window=window/2;
    int sum=0;
    int index=0;
    for (int row=window;row<(N-window); row++){
        for(int col =window; col<(N-window);col++){
            index=col + row * N;
            sum=0;
            for(int x=-window;x<=window;x++){
                for(int y=-window;y<=window;y++){
                    sum += inputImage[index + N*x + y];
                }
            }
            outputImage[index] = sum/((window*2+1)*(window*2+1));
        }
    }
}



int main(int argc, char**argv)
{
    
    const int N=strtol(argv[1],NULL, 10);
    const int window=strtol(argv[2],NULL,10);
    
    // Input image and output image for the testing
    int *inputImgageS, *outputImgageS,*outputImgageS1;
    
    // Input and output image for the filering 
    int *inputImage, *outputImage;
    int *inputImage_h, *outputImage_h;
    
    // Number of elements in a 2D array image
    int size = N * N * sizeof(int);

    // Allcate memory for input output image 
    inputImgageS= (int*)malloc(size);
    outputImgageS= (int*)malloc(size);
    outputImgageS1=(int*)malloc(size);

    
    // Create a image5(2D Array) with random numbers
    for (int row=0; row<N; row++) {
        for (int col=0; col<N; col++){
            inputImgageS[col + row * N]=rand() % 256;
            // inputImgageS[row][col] = rand() % 256;
        }
    }
    

    // Allocate memory on the GPUs for the image
    cudaMalloc((void**)&inputImage, size);
    cudaMalloc((void**)&outputImage, size);

    //Allocate memory on the GPUs for the image
    inputImage_h= (int*)malloc(size);
    outputImage_h= (int*)malloc(size);

    // Copy the image form host to device (input and output)
    cudaMemcpy(inputImage, inputImgageS, size, cudaMemcpyHostToDevice);
    cudaMemcpy(outputImage, inputImgageS, size, cudaMemcpyHostToDevice);

    // Copy input image into output image 
    cudaMemcpy(inputImage_h,inputImgageS,size,cudaMemcpyHostToHost);
    cudaMemcpy(outputImage_h,inputImgageS,size,cudaMemcpyHostToHost); 

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
    clock_t start_h = clock();
    printf("Doing CPU mean filter\n");
    mean_Filter_h(inputImage_h,outputImage_h,window,N);
    clock_t end_h = clock();

    //  Time calculation
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    // Copy GPUs' output image to Host 
    cudaMemcpy(outputImgageS, outputImage , size, cudaMemcpyDeviceToHost);
    // Cpoy CPUs' output image to host 
    cudaMemcpy(outputImgageS1, outputImage_h , size, cudaMemcpyHostToHost);

    printf("Image size : %d Window size : %d GPU Time: %f CPU Time: %f\n",N,window,time_d,time_h);
    // printf("Image size : %d Window size : %d GPU Time: %f \n",N,window,time_d);
    
    // // Print Imput image
    // printf("Input image \n");
    // for (int row=0;row< N;row++){
    //     printf("[");
    //     for(int col=0;col<N;col++){
    //         printf("%d,",inputImgageS[row][col]);
    //     }
    //     printf("]\n");
    // }

    // // Print output image 
    // printf("Output Image\n");
    // for (int row=0;row< N;row++){
    //     printf("[");
    //     for(int col=0;col<N;col++){
    //         printf("%d,",outputImgageS[row][col]);
    //     }
    //     printf("]\n");
    // }

    // Check both output matrix are same
    printf("Both outputs are same : ");
    bool check=true;
    for (int row=0;row< N;row++){
        for(int col=0;col<N;col++){
            if(outputImgageS[row*N+col]!=outputImgageS1[row*N+col]){
                check=false;
            }else{
                check=true;
            }
        }
    }
    // Print the status 
    if(check){
        printf("YES\n");
    }else{
        printf("NO\n");
    }


    // Free memory on GPU 
    cudaFree(inputImage);
    cudaFree(outputImage);

    // Free host memory
    free(inputImgageS);
    free(outputImgageS);
    free(outputImgageS1);
    free(inputImage_h);
    free(outputImage_h);

    return 0;
    
}