from onnxscript.tensor import Tensor
import numpy as np
import onnxruntime as ort
import time
import matplotlib.pyplot as plt
import cupy as cp


def test_onnx_gpu_speed():
    # you need to uninstall onnxruntime and install onnxruntime-gpu to run this test

    methods = ["onnx gpu", "onnx cpu", "numpy", "cupy"]
    min_durations: dict[str, list[float]] = {}
    num_elements=[n*n for n in  [10,100,500,1000,2000,5000]]
    for n in num_elements:       
        np_array = np.ones((n), dtype=np.float32) 
        a: Tensor|np.ndarray 
        for method in methods:
            if method == "onnx gpu": 
                a = Tensor(np_array, device_type="cuda", device_id=0)
            elif method == "onnx cpu":
                a = Tensor(np_array, device_type="cpu", device_id=0)
            elif method == "numpy":
                a = np_array
            elif method == "cupy":
                a = cp.asarray(np_array)
            else:
                raise ValueError(f"Unknown method {method}")
            durations_ms = []
            for _ in range(10):        
                start=time.perf_counter()
                b=a*2
                durations_ms.append((time.perf_counter()-start)*1000)
            if method not in min_durations:
                min_durations[method] = []
            min_durations[method].append(min(durations_ms))
            print(f"{method} min duration = {min(durations_ms)} ms")
    for method in methods:
        plt.loglog(num_elements,min_durations[method],label=method)
    plt.xlabel("number of elements")
    plt.ylabel("duration (ms)")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    test_onnx_gpu_speed()