# Train Convolutional Neural Network on MNIST Dataset

## Command-line arguments
The command-line arguments are defined in the table:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--block-size` | `int` | `1024` | Number of training samples per process. |
| `--shard-size` | `int` | `128` | Shard size used in matrix computations. |
| `--slice-size` | `int` | `32` | Slice size used for Jacobian evaluation. |
| `--damp-ratio` | `float` | `10.0` | Change ratio of the damping factor. |
| `--epochs` | `int` | `10` | Number of training epochs. |
| `--test-block-size` | `int` | `1024` | Input block size used during testing. |
| `--profile` | flag | disabled | Enable profiling mode. |
| `--profile-steps` | `int` | `5` | Number of training steps to run before exiting profiling mode. |
| `--test-interval` | `int` | `1` | Number of epochs between test runs. |
| `--data-dir` | `str` | `.data` | Directory containing the MNIST dataset. |
| `--log` | `str` | `None` | Path to the log file. |


## Result
Here is an example of training on a single node of 2 NVIDIA Tesla V100 PCIE (32GB) GPUs:
```bash
torchrun --standalone --nproc_per_node 2 experiments/mnist/train.py \
    --block-size 4096 \
    --shard-size 1024 \
    --slice-size 256 \
    --epochs 1
```

The output result is:
```
2026-08-03 06:31:14.625 | Model size: 1,199,882
2026-08-03 06:31:14.626 | Batch size: 8,192
2026-08-03 06:31:14.626 | Hessian memory: 0.268 GB
2026-08-03 06:31:14.627 | Jacobian shard memory: 4.915 GB
2026-08-03 06:31:32.080 | step info: epoch: 1 | batch: [1, 8192] | loss: 1.954e+00 | method: underdetermined | iterations: 6 | damp: 1.000e+01
2026-08-03 06:31:43.451 | step info: epoch: 1 | batch: [8193, 16384] | loss: 1.493e+00 | method: underdetermined | iterations: 1 | damp: 1.000e+00
2026-08-03 06:31:54.963 | step info: epoch: 1 | batch: [16385, 24576] | loss: 1.233e+00 | method: underdetermined | iterations: 2 | damp: 1.000e+00
2026-08-03 06:32:06.541 | step info: epoch: 1 | batch: [24577, 32768] | loss: 6.044e-01 | method: underdetermined | iterations: 3 | damp: 1.000e+01
2026-08-03 06:32:18.531 | step info: epoch: 1 | batch: [32769, 40960] | loss: 3.803e-01 | method: underdetermined | iterations: 1 | damp: 1.000e+00
2026-08-03 06:32:30.775 | step info: epoch: 1 | batch: [40961, 49152] | loss: 1.972e-01 | method: underdetermined | iterations: 1 | damp: 1.000e-01
2026-08-03 06:32:43.285 | step info: epoch: 1 | batch: [49153, 57344] | loss: 8.006e-02 | method: underdetermined | iterations: 1 | damp: 1.000e-02
2026-08-03 06:32:44.917 | step info: epoch: 1 | batch: [57345, 60000] | loss: 1.559e-02 | method: underdetermined | iterations: 1 | damp: 1.000e-03
2026-08-03 06:32:44.919 | epoch info | epoch: 1 | loss: 5.958e+00 | memory: 20.12 GB
2026-08-03 06:32:45.870 | test info | epoch: 1 | loss: 6.147e-02 | accuracy: 98.45%
```

