NPROC=${NPROC:-1}
torchrun --standalone --nproc_per_node $NPROC experiments/mnist/train.py "$@"
