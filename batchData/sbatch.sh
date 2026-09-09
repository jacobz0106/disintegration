#!/bin/bash
# Submit data-generation jobs for all four examples across both sampling
# methods. Each per-job script (generateData.py under the hood) skips
# (n, interval, repeat) combos whose dQ_Train CSV already exists, so re-running
# this driver only fills in gaps.
examples=("function2" "brusselator" "lotka" "SIR")
samples=(20 10 5)
methods=("Random" "POF")

for example in "${examples[@]}"; do
  for size in "${samples[@]}"; do
    for method in "${methods[@]}"; do
      jobname="${example}_${size}_${method}"
      sbatch --time=6-23:00:00 \
             --cpus-per-task=5 \
             --mem-per-cpu=5G \
             -o out/${jobname}.out \
             -e out/${jobname}.err \
             ${jobname}.sh
    done
  done
done
