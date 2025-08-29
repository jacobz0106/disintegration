#!/bin/bash
#!/bin/bash
models=("PPSVMG" "NN")
samples=(20 10 5)
methods=("Random" "POF")

for model in "${models[@]}"; do
  for size in "${samples[@]}"; do
    for method in "${methods[@]}"; do
      jobname="lotka_${model}_${size}_${method}"
      sbatch --time=7-00:00:00 \
             --cpus-per-task=5 \
             --mem-per-cpu=10G \
             -o out/${jobname}.out \
             -e out/${jobname}.err \
             ${jobname}.sh
    done
  done
done
