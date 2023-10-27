import torch
import os
import re


def is_torch_sub_class(obj):
    for parent in obj.__class__.__mro__:
        if parent.__module__.startswith("torch"):
            return True
    return False


# noinspection PyBroadException
def find_tensors(obj, obj_path, results, depth, max_depth=5):
    if depth > max_depth or obj == results:
        return
    if isinstance(obj, (list, tuple, set)):
        for i, o in enumerate(obj):
            find_tensors(o, f"{obj_path}[{i}]", results, depth + 1, max_depth)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_tensors(v, f"{obj_path}[{k}]", results, depth + 1, max_depth)

    if type(obj) is torch.Tensor:
        results.setdefault(obj_path, obj)
    elif is_torch_sub_class(obj):
        for attrName in dir(obj):
            try:
                find_tensors(getattr(obj, attrName), f"{obj_path}.{attrName}", results, depth + 1, max_depth)
            except Exception as _:
                pass


# noinspection PyBroadException
def is_tensor(obj):
    try:
        return isinstance(obj, torch.Tensor) or (hasattr(obj, "data") and isinstance(obj.data, torch.Tensor))
    except Exception as _:
        return False


def output_tensor_summary(deep_traverse=False):
    from gc import get_objects
    from warnings import filterwarnings
    from collections import Counter

    unit, unit_name = 1024, "KB"

    filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")

    if deep_traverse:
        global_tensors = {}
        find_tensors(globals().copy(), "Global", global_tensors, 0)
        global_tensors = {id(v): k for k, v in global_tensors.items()}
    else:
        global_tensors = {id(v): k for k, v in globals().items() if is_tensor(v)}

    total_usage = 0
    trivial_memory_usage = 0
    big_tensors = []
    for obj in get_objects():
        try:
            if is_tensor(obj):
                if obj.device.index is None:
                    continue
                tensor_mem_size = obj.nelement() * obj.element_size()
                total_usage += tensor_mem_size
                if (tensor_mem_size / unit) < 1:
                    trivial_memory_usage += tensor_mem_size
                    continue
                if id(obj) in global_tensors:
                    big_tensors.append((obj.shape, tensor_mem_size / unit, global_tensors[id(obj)]))
                else:
                    big_tensors.append((obj.shape, tensor_mem_size / unit))
        finally:
            pass

    print(f"Total {total_usage / unit:.2f} {unit_name} CUDA memory in use.\n")

    big_tensors.sort(key=lambda x: x[1], reverse=True)

    max_lower_unit, min_lower_unit = 1000, 100
    while min_lower_unit >= 1:
        in_range_tensors = [t for t in big_tensors if min_lower_unit <= t[1] <= max_lower_unit]
        group_counter = Counter(in_range_tensors)
        print(f"Tensors of size {min_lower_unit:>5} - {max_lower_unit:>5} {unit_name}:")
        for tensor, count in group_counter.items():
            print(f"  {count:4} * Size: {tensor[1]:.2f} {unit_name} Shape: {[*tensor[0]]}", end="", )
            print(f' {tensor[2]:.30}' if len(tensor) == 3 else "")

        print(f"Total: {sum([t[1] for t in in_range_tensors]):.2f} {unit_name}\n")
        max_lower_unit, min_lower_unit = max_lower_unit // 10, min_lower_unit // 10

    print(f"Total {trivial_memory_usage / unit :.2f} {unit_name} is occupied by trivial tensors(<=1{unit_name}).")


def nvidia_smi_stat():
    expr = re.compile(r'\|\s*(\d)+\s*.+?\s*\|.+?\|.+?\|\n\|\s*(\d+)%\s*(\d+).+?(\d+)W / (\d+)W \|\s*(\d+)MiB\s/\s*(\d+)MiB\s*\|\s*(\d+)%.+?\|')
    stat = os.popen('nvidia-smi').read()
    return {i: {'id': int(v[0]), 'fan': int(v[1]), 'temperature': int(v[2]), 'power': int(v[3]), 'max_power': int(v[4]),
                'mem_usage': int(v[5]), 'mem_capacity': int(v[6]), 'util': int(v[7])}
            for i, v in enumerate(expr.findall(stat))}
