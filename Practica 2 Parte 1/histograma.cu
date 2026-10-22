#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// Esta funcion saca el histograma en CPU de manera normal (uno por uno)
void computeHistogramCPU(uint32_t *histogram, const uint8_t *image, int n) {
    for (int i = 0; i < n * n; i++) {
        histogram[image[i]]++;
    }
}

// Kernel en GPU usando memoria compartida para que vaya mas rapido
__global__ void computeHistogramShared(uint32_t *d_histogram, const uint8_t *d_image, const int n) {
    __shared__ uint32_t hist_local[256];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < 256) {
        hist_local[tid] = 0;
    }
    __syncthreads();

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < n && iy < n) {
        uint8_t pixelValue = d_image[iy * n + ix];
        atomicAdd(&hist_local[pixelValue], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&d_histogram[tid], hist_local[tid]);
    }
}

int main(int argc, char **argv) {
    uint32_t *histogram_gpu, *histogram_cpu, *d_histogram;
    uint8_t *image, *d_image;
    int n;

    if (argc != 2) {
        fprintf(stderr, "Uso: %s <tamaño_de_la_imagen>\n", argv[0]);
        return 1;
    }
    n = atoi(argv[1]);

    // Reservamos memoria (host)
    histogram_gpu = (uint32_t *)calloc(256, sizeof(uint32_t));
    histogram_cpu = (uint32_t *)calloc(256, sizeof(uint32_t));
    image = (uint8_t *)malloc(n * n * sizeof(uint8_t));

    // Llenamos la imagen con numeros aleatorios entre 0 y 255
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        image[i] = rand() % 256; 
    }

    // Eventos de CUDA para medir tiempos
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    // Tiempo en CPU
    
    clock_t start_cpu = clock();
    computeHistogramCPU(histogram_cpu, image, n);
    clock_t stop_cpu = clock();
    double time_cpu = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0; // lo pasamos a ms

   
    // Tiempo en GPU (incluyendo copias de memoria)
    
    cudaEventRecord(start); // arrancamos cronometro

    cudaMalloc((void**)&d_histogram, 256 * sizeof(uint32_t));
    cudaMalloc((void**)&d_image, n * n * sizeof(uint8_t));

    cudaMemcpy(d_histogram, histogram_gpu, 256 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image, n * n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Puedes probar BLOCK_SIZE en 8, 16 o 32 para comparar
    int BLOCK_SIZE = 16;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    
    // Asi calculamos el grid para que funcione con cualquier n
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    computeHistogramShared<<<gridSize, blockSize>>>(d_histogram, d_image, n);

    cudaMemcpy(histogram_gpu, d_histogram, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // esperamos a que termine todo
    
    float time_gpu_total = 0;
    cudaEventElapsedTime(&time_gpu_total, start, stop);

    
    // Tiempo en GPU (solo kernel)
    
    // Dejamos histograma en cero para medir limpio
    cudaMemset(d_histogram, 0, 256 * sizeof(uint32_t));
    
    cudaEventRecord(start);
    computeHistogramShared<<<gridSize, blockSize>>>(d_histogram, d_image, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_gpu_kernel = 0;
    cudaEventElapsedTime(&time_gpu_kernel, start, stop);

    // Mostramos resultados
    printf("Tamaño de imagen: %dx%d\n", n, n);
    printf("Tamaño de bloque: %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("----------------------------------------\n");
    printf("Tiempo CPU: %.3f ms\n", time_cpu);
    printf("Tiempo GPU (Total): %.3f ms\n", time_gpu_total);
    printf("Tiempo GPU (Solo Kernel): %.3f ms\n", time_gpu_kernel);
    printf("----------------------------------------\n");
    printf("Aceleración (Speedup) Total: %.2fx\n", time_cpu / time_gpu_total);
    printf("Aceleración (Speedup) Solo Kernel: %.2fx\n", time_cpu / time_gpu_kernel);

    // Liberamos memoria al final
    cudaFree(d_histogram);
    cudaFree(d_image);
    free(histogram_gpu);
    free(histogram_cpu);
    free(image);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
