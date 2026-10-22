#include <iostream>
#include <vector>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 3840, height = 2160;
    int channels = 3;
    int max_iter = 1000;

    int pixeles_totales = width * height;
    
    // si no se puede partir bien, mejor parar aqui
    if (pixeles_totales % size != 0) {
        if (rank == 0) std::cerr << "Error: Píxeles no divisibles entre procesos." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int pixeles_locales = pixeles_totales / size;
    int datos_locales = pixeles_locales * channels; // 3 bytes por píxel (RGB)

    std::vector<unsigned char> buffer_local(datos_locales);
    unsigned char* img_resultado = nullptr;

    if (rank == 0) {
        img_resultado = new unsigned char[pixeles_totales * channels];
        std::cout << "Generando fractal en 4K (Paralelo) con " << size << " procesos..." << std::endl;
    }

    // aqui empieza el tiempo del paralelo
    double inicio = MPI_Wtime();

    // cada proceso calcula su trozo de la imagen
    for (int i = 0; i < pixeles_locales; i++) {
        // ver que pixel global me toca segun mi rank
        int pixel_global = (rank * pixeles_locales) + i;
        
        int x = pixel_global % width;
        int y = pixel_global / width;

        float cx = -2.5f + 3.5f * (float)x / width;
        float cy = -1.0f + 2.0f * (float)y / height;

        float zx = 0.0f, zy = 0.0f;
        int iter = 0;

        while (zx * zx + zy * zy < 4.0f && iter < max_iter) {
            float temp = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = temp;
            iter++;
        }

        int idx = i * 3; // indice dentro del buffer local
        if (iter == max_iter) {
            buffer_local[idx] = 0; buffer_local[idx+1] = 0; buffer_local[idx+2] = 0;
        } else {
            buffer_local[idx]   = (iter * 5) % 256;
            buffer_local[idx+1] = (iter * 7) % 256;
            buffer_local[idx+2] = (iter * 11) % 256;
        }
    }

    // el proceso 0 recoge todo lo que calcularon los demas
    MPI_Gather(buffer_local.data(), datos_locales, MPI_UNSIGNED_CHAR,
               img_resultado, datos_locales, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // aqui acaba el tiempo del paralelo
    double fin = MPI_Wtime();

    if (rank == 0) {
        stbi_write_jpg("paralelo_mandelbrot.jpg", width, height, channels, img_resultado, 100);
        delete[] img_resultado;
        std::cout << "Tiempo paralelo: " << (fin - inicio) << " segundos" << std::endl;
    }

    MPI_Finalize();
    return 0;
}