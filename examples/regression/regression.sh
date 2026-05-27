export OMPI_ALLOW_RUN_AS_ROOT=1 
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

unset UCX_IB_REG_METHODS
unset IB_SEG_SIZE
unset UCX_IB_SEG_SIZE

legate --nodes 1 --ranks-per-node 2 --cpus 1 --gpus 1 \
      --fbmem 2048 --sysmem 512 --min-gpu-chunk 1 --gpu-bind 0/1 \
      --launcher mpirun examples/regression/regression.py --driver legate
