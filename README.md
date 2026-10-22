# Parallel Computing with MPI and CUDA

Three exercises from my High Performance Computing course, covering distributed CPU workloads and GPU memory optimization. Each exercise includes the source code and a report with the implementation decisions and performance measurements.

## Projects

| Exercise | Implementation | Main idea |
| --- | --- | --- |
| Mandelbrot renderer | C++ and MPI | Split a 4K image between processes and gather the results on the root process. |
| Image histogram | CUDA | Accumulate a 256-bin histogram in shared memory before merging it into global memory. |
| Dense layer inference | CUDA | Compare direct global-memory access with shared-memory tiling for matrix multiplication followed by ReLU. |

### Mandelbrot renderer

The sequential and MPI versions generate the same 3840 × 2160 fractal with a limit of 1,000 iterations per pixel. The MPI version assigns a contiguous range of pixels to each process and combines the RGB buffers with `MPI_Gather`.

Both programs save a JPEG and print the calculation time. Image encoding is outside the timed section; the parallel measurement includes gathering the pixel buffers.

### Image histogram

The program generates a square grayscale image and calculates its histogram on both the CPU and GPU. The CUDA kernel uses one shared histogram per block, atomic updates and synchronization before merging the partial histograms.

It prints the CPU time, a GPU timing that includes memory operations, and a separate kernel-only timing. The default block size is 16 × 16.

### Dense layer inference

Two kernels compute `Y = ReLU(X × W)`: one reads directly from global memory, while the other reuses 16 × 16 tiles in shared memory. CUDA events measure the execution time of each kernel and the program prints their time ratio.

The default dimensions are 10,000 input samples, 1,024 features and 2,048 output neurons. Inputs and weights are generated in the program; no dataset or trained model is required.

## Build and run

Requirements depend on the exercise:

- A C++11 compiler for the sequential renderer.
- An MPI implementation providing `mpicxx` and `mpirun` for the parallel renderer.
- The NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable NVIDIA GPU for the CUDA exercises.
- Slurm only if using the included cluster submission script.

Run the following commands from the repository root.

**Mandelbrot**

```bash
cd "Practica 1"
g++ -std=c++11 -O3 mandelbrot_secuencial.cpp -o secuencial
mpicxx -std=c++11 -O3 mandelbrot_paralelo.cpp -o paralelo
./secuencial
mpirun -np 4 ./paralelo
```

The MPI process count must divide the total number of pixels (8,294,400). The output files are `secuencial_mandelbrot.jpg` and `paralelo_mandelbrot.jpg` in the current directory.

For a Slurm cluster, adjust the partition, node count and tasks per node in `mandelbrot.sh`, then submit it from `Practica 1` with `sbatch mandelbrot.sh`.

**Histogram**

```bash
cd "Practica 2 Parte 1"
nvcc -O3 histograma.cu -o histograma
./histograma 4096
```

The argument is the image side length: `4096` generates a 4096 × 4096 image. Keep the default 16 × 16 block size when using the code as provided.

**Dense layer inference**

```bash
cd "Practica 2 Parte 2"
nvcc -O3 inferencia_optimizada.cu -o inferencia
./inferencia
```

Problem dimensions and tile size are defined in the source files. Timing results depend on the hardware and configuration.

## Reports

The original reports are in Spanish and include the experiments, tables and discussion:

- [Mandelbrot and MPI](Practica%201/mandelbrot.pdf)
- [CUDA histogram](Practica%202%20Parte%201/Acap%20p2p1.pdf)
- [CUDA dense layer inference](Practica%202%20Parte%202/Acap%20p2p2.pdf)

The renderer includes `stb_image_write.h` from the [stb libraries](https://github.com/nothings/stb) for JPEG output. Its license is included in the header.
