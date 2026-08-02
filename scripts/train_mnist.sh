export OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node 2 experiments/mnist/train.py $@