# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
export SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_PATH

mkdir -p logs

for NENV in 2 4 8 16 32 64 128 356 512 1024 2048 4096 8192 16384
do
    python test_gym.py --task=Door --num_envs=${NENV} > logs/output_n_envs_${NENV}.txt
done
