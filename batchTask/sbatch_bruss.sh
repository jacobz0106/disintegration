#!/bin/bash
#!/bin/bash
models=("NN")
samples=(20 10 5)
methods=("Random" "POF")

for model in "${models[@]}"; do
  for size in "${samples[@]}"; do
    for method in "${methods[@]}"; do
      jobname="brusselator_${model}_${size}_${method}"
      sbatch --time=13-00:00:00 \
             --cpus-per-task=5 \
             --mem-per-cpu=5G \
             -o out/${jobname}.out \
             -e out/${jobname}.err \
             ${jobname}.sh
    done
  done
done
