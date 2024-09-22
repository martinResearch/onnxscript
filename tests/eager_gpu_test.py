from onnxscript.tensor import Tensor
import numpy as np
import onnxruntime as ort

def test():
    np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # you need to uninstall onnxruntime and install onnxruntime-gpu
    ort_value = ort.OrtValue.ortvalue_from_numpy(np_array)
    a= Tensor(ort_value)
    for i in range(10):
        b=a+a
        print(b)

if __name__ == '__main__':
    test()