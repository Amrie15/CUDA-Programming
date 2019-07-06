# CUDA-Vector-Addition
Run vecadd.cu
    Compile : nvcc vecadd.cu -o vecadd
    Execute the program : ./vecadd

If there is a warning like "Architectures are deprecated" then use (Please be mindfull about the architecture version) 
    nvcc -arch=sm_30 vecadd.cu -o vecadd