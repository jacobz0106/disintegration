#!/bin/bash

# Define parameters
examples=("function2" "brusselator" "lotka")
numbers=(5 10 20)
tags=("POF" "Random")

# Loop through all combinations
for example in "${examples[@]}"; do
  for number in "${numbers[@]}"; do
    for tag in "${tags[@]}"; do

      filename="${example}_${number}_${tag}.sh"

      cat << EOF > "$filename"
#!/bin/bash
module load StdEnv/2020
module load gurobi/10.0.3 python/3.9
source ~/env_gurobi/bin/activate
pip install --no-index --upgrade pip
pip install -r ../requirements.txt
export OMP_NUM_THREADS=1
export GRB_TOKENSERVER=license1.computecanada.ca 
echo "installation of req packages done."
python ../generateData.py ${example} ${number} ${tag}
EOF

      chmod +x "$filename"
      echo "✅ Created $filename"

    done
  done
done
