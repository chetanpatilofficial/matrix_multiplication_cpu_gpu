#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#define BLOCK_SIZE 8


__global__ void matrix_multiplication_kernel(int dim_m, int dim_n, int dim_k, const int* L_matrix, const int* R_matrix, int* Res_matrix)
{
    int globx = blockIdx.x * blockDim.x + threadIdx.x;
    int globy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int l;

    for (l = 0; l < dim_n; l++)
        Res_matrix[globx * dim_k + globy] += L_matrix[globx * dim_n + l] * R_matrix[l * dim_k + globy];
}

void matrix_multiplication_cpu(int dim_m, int dim_n, int dim_k, const int* L_matrix, const int* R_matrix, int* Res_matrix)
{
    for (int i = 0; i < dim_m; ++i)
    {
        for (int j = 0; j < dim_k; ++j)
        {
            Res_matrix[i * dim_k + j] = 0;

            for (int l = 0; l < dim_n; ++l)
                Res_matrix[i * dim_k + j] += L_matrix[i * dim_n + l] * R_matrix[l * dim_k + j];
        }
    }
}

void matrix_multiplication_gpu(int dim_m, int dim_n, int dim_k, const int* L_matrix, const int* R_matrix, int* Res_matrix)
{
    dim3 dim_Grid(dim_m / BLOCK_SIZE, dim_k / BLOCK_SIZE);
    dim3 dim_Block(BLOCK_SIZE, BLOCK_SIZE);
    matrix_multiplication_kernel << <dim_Grid, dim_Block >> > (dim_m, dim_n, dim_k, L_matrix, R_matrix, Res_matrix);
}

void print_matrices(int* matrix, char* file_Name, int x_dim, int y_dim, int dim)
{
    std::ofstream outFile;
    outFile.open(file_Name);

    outFile << std::fixed;
    outFile << std::setprecision(2);

    for (int i = 0; i < x_dim; i++) {

        for (int j = 0; j < y_dim; j++) {
            outFile << matrix[i * dim + j] << " ";
            //printf("%d ", matrix[i * dim + j]);
        }
        //printf("\n");
        outFile << std::endl;
    }
}

int main()
{

    printf("Enter dim_m, dim_n, dim_k values:\n");
    int m, n, k;
    scanf("%d %d %d", &m, &n, &k);
    const int dim_m = m, dim_n = n, dim_k = k;
    int* L_matrix, * R_matrix, * Res_matrix, * L_matrix_gpu, * R_matrix_gpu, * Res_matrix_gpu, * c_verify, * Res_host;

    L_matrix = new int[dim_m * dim_n];
    R_matrix = new int[dim_n * dim_k];
    Res_matrix = new int[dim_m * dim_k];

    cudaMalloc((void**)&L_matrix_gpu, dim_m * dim_n * sizeof * L_matrix_gpu);
    cudaMalloc((void**)&R_matrix_gpu, dim_n * dim_k * sizeof * R_matrix_gpu);
    cudaMalloc((void**)&Res_matrix_gpu, dim_m * dim_k * sizeof * Res_matrix_gpu);

    //fill matrices with 1's and 0's
    for (int i = 0; i < dim_m * dim_n; ++i)
        L_matrix[i] = (i % 2 == 0 ? 1 : 0);

    for (int i = 0; i < dim_n * dim_k; ++i)
        R_matrix[i] = (i % 2 == 0 ? 1 : 0);

    //print_matrices(L_matrix, "Input_LHS", dim_m, dim_n, dim_k);
    //print_matrices(R_matrix, "Input_RHS", dim_m, dim_n, dim_k);
    size_t vector_size;
    vector_size = dim_m * dim_k * sizeof(int);
    Res_host = (int*)malloc(vector_size);


    cudaMemcpy(L_matrix_gpu, L_matrix, dim_m * dim_n * sizeof * L_matrix_gpu, cudaMemcpyHostToDevice);
    cudaMemcpy(R_matrix_gpu, R_matrix, dim_n * dim_k * sizeof * R_matrix_gpu, cudaMemcpyHostToDevice);

    //Matrix Multiplication in host(CPU)
    float sTime;
    clock_t start, finish;
    start = clock();
    matrix_multiplication_cpu(dim_m, dim_n, dim_k, L_matrix, R_matrix, Res_matrix);
    finish = clock();

    sTime = (double)1000 * (finish - start) / CLOCKS_PER_SEC;

    //print_matrices(Res_matrix, "CPU_out", dim_m, dim_n, dim_k);
    printf("Run time on CPU: %lf ms", sTime);

    //Matrix Multiplication in device(GPU)
    start = clock();
    matrix_multiplication_gpu(dim_m, dim_n, dim_k, L_matrix_gpu, R_matrix_gpu, Res_matrix_gpu);
    cudaThreadSynchronize();
    cudaMemcpy(Res_host, Res_matrix_gpu, dim_m * dim_k * sizeof * Res_matrix_gpu, cudaMemcpyDeviceToHost);
    finish = clock();
    //print_matrices(Res_host, "GPU_out", dim_m, dim_n, dim_k);
    sTime = (double)1000 * (finish - start) / CLOCKS_PER_SEC;
    printf("Run time on GPU: %lf ms", sTime);

    cudaMemset(Res_matrix_gpu, 0, dim_m * dim_k * sizeof * Res_matrix_gpu);
    cudaFree(Res_matrix_gpu);
    cudaFree(L_matrix_gpu);
    cudaFree(R_matrix_gpu);

    delete[] Res_matrix;
    delete[] R_matrix;
    delete[] L_matrix;
}