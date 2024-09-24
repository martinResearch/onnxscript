# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numbers
from typing import Optional, Sequence

import numpy as np
import onnx
import onnx.helper

from onnxscript import tensor
import onnxruntime as ort


def external_tensor(
    name: str,
    data_type: int,
    dims: Sequence[int],
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> onnx.TensorProto:
    """Create a TensorProto referencing externally stored tensor-data.

    Args:
        name: name of the tensor
        data_type: data type of tensor element
        dims: shape of the tensor
        location: location of the external file (relative path)
        offset: offset in the file where the tensor-data starts
        length: number of bytes containing the data
        checksum: SHA1 digest of the file
        basepath: basepath combined with location to form the full path

    Returns:
        TensorProto

    See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md for more details.
    """
    tensor_proto = onnx.TensorProto()
    tensor_proto.name = name
    tensor_proto.data_type = data_type
    tensor_proto.dims.extend(dims)
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL

    def add(k, v):
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)

    add("location", location)
    if offset is not None:
        add("offset", int(offset))
    if length is not None:
        add("length", int(length))
    if checksum is not None:
        add("checksum", checksum)
    if basepath is not None:
        add("basepath", basepath)
    return tensor_proto


ort_type_to_np_dtype_types_map = {
    "tensor(float)": np.dtype("float32"),
    "tensor(double)": np.dtype("float64"),
    "tensor(int32)": np.dtype("int32"),
    "tensor(int64)": np.dtype("int64"),
    "tensor(bool)": np.dtype("bool"),
    "tensor(uint8)": np.dtype("uint8"),
    "tensor(uint16)": np.dtype("uint16"),
    "tensor(int8)": np.dtype("int8"),
    "tensor(int16)": np.dtype("int16"),
    "tensor(complex64)": np.dtype("complex64"),
    "tensor(complex128)": np.dtype("complex128"),
    "tensor(float16)": np.dtype("float16"),
}


def ort_type_to_np_dtype(ort_type: str) -> np.dtype:
    return ort_type_to_np_dtype_types_map[ort_type]

ort_type_to_tensor_dtype_map={
    "tensor(float)": onnx.TensorProto.FLOAT,
    "tensor(double)": onnx.TensorProto.DOUBLE,
    "tensor(int32)": onnx.TensorProto.INT32,
    "tensor(int64)": onnx.TensorProto.INT64,
    "tensor(bool)": onnx.TensorProto.BOOL,
    "tensor(uint8)": onnx.TensorProto.UINT8,
    "tensor(uint16)": onnx.TensorProto.UINT16,
    "tensor(int8)": onnx.TensorProto.INT8,
    "tensor(int16)": onnx.TensorProto.INT16,
    "tensor(complex64)": onnx.TensorProto.COMPLEX64,
    "tensor(complex128)": onnx.TensorProto.COMPLEX128,
    "tensor(float16)": onnx.TensorProto.FLOAT16,
}

def value_to_type_proto(val):
    """Return the ONNX type of a python-value."""
    if isinstance(val, (np.ndarray, tensor.Tensor)):
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(val.dtype)
        shape = val.shape
        return onnx.helper.make_tensor_type_proto(elem_type, shape)
    if isinstance(val, ort.OrtValue):
        elem_type = ort_type_to_tensor_dtype_map[val.data_type()]
        shape = val.shape()
        return onnx.helper.make_tensor_type_proto(elem_type, shape)
    if isinstance(val, int):
        return onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT32, [])
    if isinstance(val, (float, np.float32)):
        return onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
    if isinstance(val, list):
        if len(val) > 0:
            return onnx.helper.make_sequence_type_proto(value_to_type_proto(val[0]))
        # Edge-case. Cannot determine a suitable ONNX type for an empty list.
        # Should be using a typed-value instead.
        # Treated as a sequence of tensors of float-type.
        return onnx.helper.make_sequence_type_proto(
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, None)
        )
    if isinstance(val, numbers.Number):
        nparray = np.array(val)
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(nparray.dtype)
        return onnx.helper.make_tensor_type_proto(elem_type, [])
    raise ValueError(f"Value of type {type(val)} is invalid as an ONNX input/output.")


def values_to_value_infos(name_values):
    """Create a list of ValueInfoProto from a list of (name, value) pairs,
    skipping any None values.
    """
    return [
        onnx.helper.make_value_info(name, value_to_type_proto(val))
        for (name, val) in name_values
        if val is not None
    ]
