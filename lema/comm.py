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
        lo_row, lo_col = dest_slice.domain().lo
        hi_row, hi_col = dest_slice.domain().hi
        hi_row, hi_col = hi_row + 1, hi_col + 1 

        dest_slice = cupy.asarray(dest_slice)

        for (src_begin, src_end), src_slice in zip(src_ranges, src_slices):
            intersect_start = max(lo_row, src_begin)
            intersect_end = min(hi_row, src_end)

            if intersect_start >= intersect_end:
                continue

            dst_local_start = intersect_start - lo_row
            dst_local_end = intersect_end - lo_row

            src_local_start = intersect_start - src_begin
            src_local_end = intersect_end - src_begin
            src_slice_gpu = cupy.asarray(src_slice)
            
            dest_slice[dst_local_start:dst_local_end, :] = \
                src_slice_gpu[src_local_start:src_local_end, lo_col:hi_col]

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
# TODO: fix sync issue
def broadcast_to_torch(src: np.ndarray, device_tensor_map: Dict[torch.device, torch.Tensor]):
    @task(variants=(VariantCode.GPU,))
    def broadcast_task(ctx: TaskContext, src_slice: InputArray):
        lo, = src_slice.domain().lo
        hi, = src_slice.domain().hi
        hi = hi + 1
        src_slice = cupy.asarray(src_slice)
        for device, dest_slice in device_tensor_map.items():
            dest_slice[lo:hi] = torch.as_tensor(src_slice[:], device=device)
    
    broadcast_task(src)

'''
Broadcast a cuPyNumeric array to torch via CPU.
'''
def broadcast_to_torch_cpu(src: np.ndarray, device_tensor_map: Dict[torch.device, torch.Tensor]):
    host_src = torch.tensor(src, device='cpu', pin_memory=True)
    for device, dest in device_tensor_map.items():
        dest[:] = host_src