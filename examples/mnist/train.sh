export OMPI_ALLOW_RUN_AS_ROOT=1 
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

legate --nodes 1 --ranks-per-node 2 --cpus 1 --gpus 1 \
      --fbmem 16000 --sysmem 512 --min-gpu-chunk 1 --gpu-bind 0/1 \
      --launcher mpirun examples/mnist/train.py "$@" 
