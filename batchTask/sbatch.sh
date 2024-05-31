#!/bin/bash
sbatch --time=90:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/brussoutput.out -e out/bruss.err bruss.sh 
sbatch --time=90:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/brussoutput.out -e out/bruss.err randbruss.sh 