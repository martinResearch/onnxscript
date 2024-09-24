from onnxscript.tensor import Tensor
import numpy as np
import onnxruntime as ort
import time
import matplotlib.pyplot as plt
import cupy as cp
import itertools
from viztracer import VizTracer

def benchmark():
    # you need to uninstall onnxruntime and install onnxruntime-gpu to run this test

    methods = ["cupy", "onnx cpu", "onnx gpu", "numpy"]
    min_durations: dict[str, list[float]] = {}
    # num_elements=[n*n for n in  [10,100,500,1000,2000,5000, 7000]]
    num_trials_out = 5
    num_trials_in = 5
    num_elements_min = 100
    num_elements_max = 1000000
    n_steps = 30
    # get n_steps values between num_elements_min and num_elements_max uniformly spaced on a log scale
    num_elements = np.logspace(
        np.log10(num_elements_min),
        np.log10(num_elements_max),
        num=n_steps,
        endpoint=True,
        base=10.0,
        dtype=int,
    )
    durations_ms = {}

    a_method = {}

    for n, method in itertools.product(num_elements, methods):
        np_array = np.ones((n), dtype=np.float32)
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
        a_method[(method, n)] = a

    for trial_out, n, method, trial_in in itertools.product(
        range(num_trials_out), num_elements, methods, range(num_trials_in)
    ):
        a = a_method[(method, n)]
        start = time.perf_counter()
        b = a * a
        duration = (time.perf_counter() - start) * 1000
        durations_ms[(trial_out, n, method, trial_in)] = duration

    # get the minimum duration for each method and number of elements
    for method in methods:
        min_durations[method] = []
        for n in num_elements:
            durations = [
                durations_ms[(trial_out, n, method, trial_in)]
                for trial_in, trial_out in itertools.product(
                    range(num_trials_out), range(num_trials_in)
                )
            ]
            min_durations[method].append(min(durations))

    for method in methods:
        plt.loglog(num_elements, min_durations[method], label=method)
    plt.xlabel("number of elements")
    plt.ylabel("duration (ms)")
    plt.legend()
    plt.show()


def test_onnx_gpu_speed():
    # you need to uninstall onnxruntime and install onnxruntime-gpu to run this test

    n = 100
    np_array = np.ones((n), dtype=np.float32)

    a = Tensor(np_array, device_type="cuda", device_id=0)
    tracer = VizTracer()
    durations = []
    for iter in range(2):
        if iter>0:
            tracer.start()
        start = time.perf_counter()
        b = a * a
        duration = (time.perf_counter() - start) * 1000
        durations.append(duration)
    tracer.stop()
    tracer.save() #
    print(min(durations))


if __name__ == "__main__":
    test_onnx_gpu_speed()
    # benchmark()
