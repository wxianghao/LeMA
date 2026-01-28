import cupynumeric as np
import cupy
import torch
from legate.core import TaskContext, VariantCode
from legate.core.task import task, InputArray, OutputArray
from typing import List, Tuple, Dict

'''
Gather row slices into dest using the specified src_ranges.

Parameters:
    src_ranges - List of [start, end) row intervals.
    src_slices - List of row-based data slices.
    dest - Output cuPyNumeric array.
'''
def gather_to_cupynumeric(src_ranges: List[Tuple[int, int]], src_slices: List[torch.Tensor], dest: np.ndarray):
    @task(variants=(VariantCode.GPU,))
    def gather_task(ctx: TaskContext, dest_slice: OutputArray):
        # Determine the destination slice
        lo_row, lo_col = dest_slice.domain().lo
        hi_row, hi_col = dest_slice.domain().hi
        hi_row, hi_col = hi_row + 1, hi_col + 1
        assert lo_row == 0 # Assume the cupynumeric array does not split the row direction

        dest_slice = cupy.asarray(dest_slice)

        # Copy loop
        for (src_begin, src_end), src_slice in zip(src_ranges, src_slices):
            src_slice = cupy.asarray(src_slice)
            dest_slice[src_begin:src_end, :] = src_slice[:, lo_col:hi_col]

    gather_task(dest)


'''
Send a torch tensor to cuPyNumeric.
    
Parameters:
    src - Tensor to be sent.
    dest - Output cuPyNumeric array.
'''   
def send_to_cupynumeric(src: torch.Tensor, dest: np.ndarray):
    @task(variants=(VariantCode.GPU,))
    def send_task(ctx: TaskContext, dest_slice: OutputArray):
        lo, = dest_slice.domain().lo
        hi, = dest_slice.domain().hi
        hi = hi + 1
        dest_slice = cupy.asarray(dest_slice)
        dest_slice[:] = cupy.asarray(src)[lo:hi]
        
    send_task(dest)

'''
Broadcast a cuPyNumeric array to torch.

Parameters:
    src - cuPyNumeric array to be sent.
    device_tensor_map - Mapping CUDA device to output torch tensor.
'''
def broadcast_to_torch(src: np.ndarray, device_tensor_map: Dict[torch.device, torch.Tensor]):
    @task(variants=(VariantCode.GPU,))
    def broadcast_task(ctx: TaskContext, src_slice: InputArray):
        lo, = src_slice.domain().lo
        hi, = src_slice.domain().hi
        hi = hi + 1
        src_slice = cupy.asarray(src_slice)
        for device, dest_slice in device_tensor_map.items():
            dest_slice[lo:hi] = torch.as_tensor(src_slice[:], device=device)
            # TODO: fix this weird issue
            # print(f"{lo}:{hi}: {dest_slice[lo:hi].sum().item()} {src_slice[:].sum().item()}")
    
    broadcast_task(src)
    # for device, x in device_tensor_map.items():
        # print(device, x)
    # print()