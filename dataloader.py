import numpy as np
from torch.utils.data import IterableDataset
from typing import *
import struct
from functools import reduce


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

    def __del__(self):
        self.file_handle.close()


class DatasetReader(IterableDataset):
    def __init__(self, map_file: str):
        self.map_file = map_file
        temp_iter = DatasetReaderIter(map_file)
        self.entry_count = temp_iter.entry_count

    def __len__(self):
        return self.entry_count

    def __iter__(self):
        return DatasetReaderIter(self.map_file)
