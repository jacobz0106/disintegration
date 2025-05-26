#!/bin/bash
#!/bin/bash
exampless=("lotka")
samples=(20 10 5)
methods=("Random" "POF")

for example in "${examples[@]}"; do
  for size in "${samples[@]}"; do
    for method in "${methods[@]}"; do
      jobname="brusselator_${example}_${size}_${method}"
      sbatch --time=13-00:00:00 \
             --cpus-per-task=5 \
             --mem-per-cpu=5G \
             -o out/${jobname}.out \
             -e out/${jobname}.err \
             ${jobname}.sh
    done
  done
done
