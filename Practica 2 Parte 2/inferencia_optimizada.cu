#include <iostream>
#include <cstdlib>

// Tamaño del bloque (tile).
const int TILE_WIDTH = 16;

// Kernel lento: memoria global pura (sin optimizar).
// cada hilo calcula un elemento de salida leyendo directo de memoria global.
__global__ void neuralInferenceGlobalKernel(const float *d_X, const float *d_W, float *d_Y, int batchSize, int features, int neurons) {
    // Coordenadas del hilo en la matriz de salida
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batchSize && col < neurons) {

        float value = 0.0f;

        // Leer directamente de d_X y d_W satura el bus de memoria
        for (int k = 0; k < features; ++k) {
            value += d_X[row * features + k] * d_W[k * neurons + col];
        }

        // ReLU simple (si es negativo, lo llevamos a 0)
        if (value < 0.0f) value = 0.0f;
        d_Y[row * neurons + col] = value;
    }
}

// Kernel rapido: memoria compartida (tiling).
// cargamos sub-bloques de X y W en shared para reutilizarlos.
__global__ void neuralInferenceSharedKernel(const float *d_X, const float *d_W, float *d_Y, int batchSize, int features, int neurons) {
    
    // Memoria compartida para un tile de X y otro de W en la memoria cache de la grafica
    __shared__ float ds_X[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_W[TILE_WIDTH][TILE_WIDTH];

    // Índices del bloque y del hilo
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Posición global en la salida
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float value = 0.0f;

    // Recorremos la dimensión 'features' por tiles
    for (int m = 0; m < (features + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (row < batchSize && (m * TILE_WIDTH + tx) < features) {
            ds_X[ty][tx] = d_X[row * features + (m * TILE_WIDTH + tx)];
        } else {
            ds_X[ty][tx] = 0.0f;
        }

        if ((m * TILE_WIDTH + ty) < features && col < neurons) {
            ds_W[ty][tx] = d_W[(m * TILE_WIDTH + ty) * neurons + col];
        } else {
            ds_W[ty][tx] = 0.0f;
        }

        // Esperamos a que todo el bloque cargue su tile en la memoria compartida
        __syncthreads();

        // Multiplicación de los tiles locales
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += ds_X[ty][k] * ds_W[k][tx];
        }

        // Sincronizamos antes de cargar el siguiente tile
        __syncthreads();
    }

    //ReLU simple
    if (row < batchSize && col < neurons) {
        if (value < 0.0f) value = 0.0f;
        d_Y[row * neurons + col] = value;
    }
}


int main() {
    // Tamaños del problema (batch, entradas y neuronas)
    int batchSize = 10000;  
    int features = 1024;
    int neurons = 2048;

    // Tamaños en bytes para reservar memoria
    size_t size_X = (size_t)batchSize * features * sizeof(float);
    size_t size_W = (size_t)features * neurons * sizeof(float);
    size_t size_Y = (size_t)batchSize * neurons * sizeof(float);

    // Memoria en CPU (host)
    float *h_X = new float[batchSize * features];
    float *h_W = new float[features * neurons];
    // Llenamos con valores aleatorios para probar
    for (int i = 0; i < batchSize * features; i++) h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < features * neurons; i++) h_W[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // Punteros en GPU
    float *d_X, *d_W, *d_Y;
    // Reservamos memoria en la GPU
    cudaMalloc((void**)&d_X, size_X);
    cudaMalloc((void**)&d_W, size_W);
    cudaMalloc((void**)&d_Y, size_Y);

    // Copiamos datos a la GPU
    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);

    // Configuración del grid y los bloques
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((neurons + TILE_WIDTH - 1) / TILE_WIDTH, (batchSize + TILE_WIDTH - 1) / TILE_WIDTH); //el -1 es para al dividir redondear hacia arriba para no dejarnos nada 

    // Eventos para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_global = 0, ms_shared = 0;

    std::cout << "test de memoria, tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;

    // Test global
    cudaEventRecord(start);
    neuralInferenceGlobalKernel<<<dimGrid, dimBlock>>>(d_X, d_W, d_Y, batchSize, features, neurons);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_global, start, stop);
    std::cout << "tiempo memoria global:      " << ms_global << " ms" << std::endl;

    // Test compartida
    cudaEventRecord(start);
    neuralInferenceSharedKernel<<<dimGrid, dimBlock>>>(d_X, d_W, d_Y, batchSize, features, neurons);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_shared, start, stop);
    
    std::cout << "tiempo memoria compartida:  " << ms_shared << " ms" << std::endl;
    
    std::cout << "aceleracion (speedup):      " << ms_global / ms_shared << "x" << std::endl;
    std::cout << "\n" << std::endl;

    // Liberamos memoria
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_Y);
    delete[] h_X; delete[] h_W;
    return 0;
}