#!/bin/bash
#SBATCH --job-name=mandelbrot_mpi
#SBATCH --partition=acap
#SBATCH --nodes=1             # <-- CAMBIAR: número de nodos (1, 2, 3)
#SBATCH --ntasks-per-node=24  # <-- CAMBIAR: procesos por nodo (8, 16, 24)
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=salida_mandelbrot_%j.txt   
#SBATCH --error=error_mandelbrot_%j.txt

# Procesos totales = --nodes * --ntasks-per-node
echo "=== Generacion Fractal Mandelbrot 4K ==="
echo "Nodos: $SLURM_NNODES  |  Procesos totales: $SLURM_NTASKS"

mpirun -np $SLURM_NTASKS ./paralelo