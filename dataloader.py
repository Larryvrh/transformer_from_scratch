import numpy as np
from torch.utils.data import IterableDataset
from typing import *
import struct
from functools import reduce
import random
import json


def json_load(path):
    with open(path, 'r') as file:
        return json.load(file)


def json_dump(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


class DatasetWriter:
    def __init__(self, map_file: str, fields: Dict[str, np.dtype]):
        self.file_handle = open(map_file, 'wb')
        self.fields = list(fields.items())
        self.fields.sort()
        self.entry_count = 0
        self.file_handle.write(self.get_header())

    @staticmethod
    def dtype_to_id(dtype: np.dtype):
        dtypes = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64]
        assert dtype in dtypes
        return dtypes.index(dtype)

    def add_entry(self, **kwargs: np.array):
        assert all(k in [n for n, d in self.fields] for k in kwargs)
        assert all(kwargs[n].dtype == d for n, d in self.fields if n in kwargs)
        entry = [kwargs[name] if name in kwargs else None for name, dtype in self.fields]
        for e in entry:
            entry_header = b''
            if e is None:
                entry_header += struct.pack('I', 0)
            else:
                entry_header += struct.pack('I', len(e.shape))
                entry_header += b''.join(struct.pack('I', d) for d in e.shape)
            self.file_handle.write(entry_header)
            if e is not None:
                self.file_handle.write(e.tobytes('C'))
        self.entry_count += 1

    def get_header(self):
        header_data = b''
        for field_name, field_dtype in self.fields:
            field_name = field_name.encode('utf-8')
            header_data += struct.pack(f'I{len(field_name)}sB', len(field_name), field_name, self.dtype_to_id(field_dtype))
        header_data = struct.pack('I', len(header_data)) + header_data
        header_data = header_data + struct.pack('Q', self.entry_count)
        return header_data

    def finish(self):
        self.file_handle.seek(0)
        self.file_handle.write(self.get_header())
        self.file_handle.close()


class DatasetReaderIter:
    def __init__(self, map_file: str):
        self.file_handle = open(map_file, 'rb')
        self.fields, self.entry_count = self.__read_header()
        self.cur_entry_index = 0

    @staticmethod
    def id_to_dtype(dtype_id: int):
        dtypes = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64]
        return dtypes[dtype_id]

    def __read_header(self):
        header_length = struct.unpack('I', self.file_handle.read(4))[0]
        header, header_cursor = self.file_handle.read(header_length), 0
        fields = []
        while header_cursor < header_length:
            name_length = struct.unpack('I', header[header_cursor:header_cursor + 4])[0]
            name, dtype_id = struct.unpack(f'{name_length}sB', header[header_cursor + 4:header_cursor + 4 + name_length + 1])
            fields.append((name.decode('utf-8'), self.id_to_dtype(dtype_id)))
            header_cursor += (4 + name_length + 1)
        entry_count = struct.unpack('Q', self.file_handle.read(8))[0]
        return fields, entry_count

    def has_next(self):
        return self.cur_entry_index < self.entry_count

    def __next__(self):
        if self.cur_entry_index == self.entry_count:
            raise StopIteration()
        entry = {}
        for field_name, field_dtype in self.fields:
            num_dims = struct.unpack('I', self.file_handle.read(4))[0]
            if num_dims == 0:
                entry[field_name] = None
                continue
            dims = struct.unpack('I' * num_dims, self.file_handle.read(4 * num_dims))
            data = np.fromfile(self.file_handle, field_dtype, reduce(lambda a, b: a * b, dims, 1)).reshape(dims)
            entry[field_name] = data
        self.cur_entry_index += 1
        return entry

    def __iter__(self):
        return self

    def get_state(self):
        return {'file_cursor': self.file_handle.tell(), 'fields': [(f[0], str(f[1])) for f in self.fields],
                'entry_count': self.entry_count, 'cur_entry_index': self.cur_entry_index}

    def set_state(self, state: Dict[str, Any]):
        assert dict([(f[0], str(f[1])) for f in self.fields]) == dict(state['fields']) and self.entry_count == state['entry_count']
        self.file_handle.seek(state['file_cursor'])
        self.cur_entry_index = state['cur_entry_index']

    def __del__(self):
        self.file_handle.close()


class DatasetReader(IterableDataset):
    def __init__(self, map_file: str):
        self.map_file = map_file
        temp_iter = DatasetReaderIter(map_file)
        self.entry_count = temp_iter.entry_count

    def __len__(self):
        return self.entry_count

    def save_iterator(self, iterator: DatasetReaderIter, path: str):
        state = iterator.get_state()
        state['map_file'] = self.map_file
        json_dump(state, path)

    def load_iterator(self, path: str) -> DatasetReaderIter:
        state = json_load(path)
        assert state.pop('map_file') == self.map_file
        iterator = iter(self)
        iterator.set_state(state)
        return iterator

    def __iter__(self) -> DatasetReaderIter:
        return DatasetReaderIter(self.map_file)


class MultiDatasetsReaderIter:
    def __init__(self, datasets: List[Tuple[DatasetReader, float]], seed: Optional[int] = None):
        self.iters = [(iter(d), w) for d, w in datasets]
        self.rng = random.Random(seed)

    def __next__(self):
        if len(self.iters) == 0:
            raise StopIteration()
        selected = self.rng.choices(self.iters, weights=[c for i, c in self.iters], k=1)[0]
        entry = next(selected[0])
        if not selected[0].has_next():
            self.iters[self.iters.index(selected)] = (selected[0], 0)
        return entry

    def get_state(self):
        iter_states = [(i.get_state(), w) for i, w in self.iters]
        rng_state = self.rng.getstate()
        return {'iter_states': iter_states, 'rng_state': rng_state}

    def set_state(self, state: Dict[str, Any]):
        for i, (iter_state, weight) in enumerate(state['iter_states']):
            assert weight == 0 or self.iters[i][1] == weight
            self.iters[i][0].set_state(iter_state)
            self.iters[i] = (self.iters[i][0], weight)
        rng_state = tuple(e if not isinstance(e, list) else tuple(e) for e in state['rng_state'])
        self.rng.setstate(rng_state)

    def __iter__(self):
        return self


class MultiDatasetsReader:
    def __init__(self, datasets: Union[List[DatasetReader], List[Tuple[DatasetReader, float]]], seed: Optional[int] = None):
        if isinstance(datasets[0], Tuple):
            self.datasets = datasets
        else:
            self.datasets = [(d, d.entry_count) for d in datasets]
        self.seed = seed

    def __len__(self):
        return sum(d.entry_count for d, c in self.datasets)

    # noinspection PyMethodMayBeStatic
    def save_iterator(self, iterator: MultiDatasetsReaderIter, path: str):
        state = iterator.get_state()
        json_dump(state, path)

    def load_iterator(self, path: str) -> MultiDatasetsReaderIter:
        state = json_load(path)
        iterator = iter(self)
        iterator.set_state(state)
        return iterator

    def __iter__(self) -> MultiDatasetsReaderIter:
        return MultiDatasetsReaderIter(self.datasets, self.seed)
