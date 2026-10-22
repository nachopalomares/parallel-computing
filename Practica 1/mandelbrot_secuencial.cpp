#include <iostream>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    // usamos una resolucion alta (4k) para que tarde un poco y se vea mejor el resultado
    int width = 3840;
    int height = 2160;
    int channels = 3; // rgb
    int max_iter = 1000; // maximo de iteraciones por pixel

    int total_pixeles = width * height;
    unsigned char* img_data = new unsigned char[total_pixeles * channels];

    std::cout << "Generando fractal en 4K (Secuencial)... paciencia." << std::endl;

    // aqui empieza el cronometro
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < total_pixeles; i++) {
        // pasar la posicion lineal i a x e y
        int x = i % width;
        int y = i / width;

        // llevar las coordenadas al plano de mandelbrot
        float cx = -2.5f + 3.5f * (float)x / width;
        float cy = -1.0f + 2.0f * (float)y / height;

        float zx = 0.0f, zy = 0.0f;
        int iter = 0;

        // el bucle pesado de mandelbrot
        while (zx * zx + zy * zy < 4.0f && iter < max_iter) {
            float temp = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = temp;
            iter++;
        }

        // poner el color segun cuanto tardo en escapar
        int idx = i * 3;
        if (iter == max_iter) {
            // no escapo: queda negro
            img_data[idx] = 0; img_data[idx+1] = 0; img_data[idx+2] = 0;
        } else {
            // escapo: le damos color segun las iteraciones
            img_data[idx]   = (iter * 5) % 256;  // rojo
            img_data[idx+1] = (iter * 7) % 256;  // verde
            img_data[idx+2] = (iter * 11) % 256; // azul
        }
    }

    // aqui acaba el cronometro
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // guardar imagen y limpiar
    stbi_write_jpg("secuencial_mandelbrot.jpg", width, height, channels, img_data, 100);
    delete[] img_data;

    std::cout << "Tiempo secuencial: " << diff.count() << " segundos" << std::endl;
    return 0;
}
