#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:05:00
#SBATCH -p batch
####SBATCH --reservation=hpcschool
module purge
module load devel/protobuf-python/3.14.0-GCCcore-10.2.0 
module load mpi/OpenMPI/4.0.5-GCC-10.2.0

python3 -m pip install --no-cache --user mpi4py
python3 -m pip install --no-cache --user numba
python3 -m pip install --no-cache --user pandas

#python3 main.py
mpirun -np $SLURM_NTASKS python3 main.py 