import tvm
from tvm.contrib import graph_runtime
import numpy as np
import time
import os

def run_model(data_shape, lib_path):
    assert os.path.isfile(lib_path)
    target = 'llvm'
    ctx = tvm.context(str(target), 0)

    loaded_lib = tvm.runtime.load_module(lib_path)
    image = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    input_data = tvm.nd.array(image.astype("float32"))
    module = graph_runtime.GraphModule(loaded_lib["default"](ctx))

    module.set_input("input0", input_data)
    start = time.time()
    for i in range(5):
        module.run()
    end = time.time() - start
    print('Opt time cost is:', end)

def run_auto_model(data_shape, lib_path, json_path, params_path):
    assert os.path.isfile(lib_path)
    assert os.path.isfile(json_path)
    assert os.path.isfile(params_path)
    target = 'llvm'
    ctx = tvm.context(str(target), 0)

    loaded_json = open(json_path).read()
    loaded_lib = tvm.runtime.load_module(lib_path)
    loaded_params = bytearray(open(params_path, "rb").read())
    image = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    input_data = tvm.nd.array(image.astype("float32"))

    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("input0", input_data)
    start = time.time()
    for i in range(5):
        module.run()
    end = time.time() - start
    print('Auto time cost is:', end)


if __name__ == '__main__':
    data_shape = (1, 1, 32, 32, 32)
    # Use the following code to test the saved .so file from from_pytorch_3d.py
    lib_path = 'opt_deploy_lib_x86.so'
    # run_model(data_shape, lib_path)

    # Use the following code to test the saved .so file from tune_relay_x86.py
    lib_path = 'auto_deploy_lib_x86.so'
    json_path = 'auto_deploy_graph_x86.json'
    params_path = 'auto_deploy_param_x86.params'
    run_auto_model(data_shape, lib_path, json_path, params_path)
