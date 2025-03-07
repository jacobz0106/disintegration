#!/bin/bash
sbatch --time=120:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/Naive.out -e out/Naive.err Naive.sh 
sbatch --time=120:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/Rand.out -e out/Rand.err ML_Rand.sh
sbatch --time=120:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/POF.out -e out/POF.err POF.sh