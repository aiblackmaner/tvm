# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tune_relay_x86:

Auto-tuning a convolutional network for x86 CPU
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Eddie Yan <https://github.com/eqy>`_

This is a tutorial about how to tune convolution neural network
for x86 CPU.
"""
import os
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime
from tvm.contrib import util
from tvm.contrib import graph_runtime

import torch
from test_3d.network import SegNet

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can either load some pre-defined network from :code:`relay.testing`
# or building :any:`relay.testing.resnet` with relay.
# We can also load models from MXNet, ONNX and TensorFlow.
#
# In this tutorial, we choose resnet-18 as tuning example.

def get_network(batch_size):
    input_shape = (batch_size, 1, 32, 32, 32)
    output_shape = (batch_size, 3, 32, 32, 32)
    model_params = torch.load('state_params.pth.tar')
    with torch.no_grad():
        net = SegNet(1, 3)
        net.eval()
        net.load_state_dict(model_params['state_dict'])
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(net, input_data).eval()
    img = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    input_name = 'input0'
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params, input_shape, output_shape


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = "llvm -mcpu=core-avx2"

batch_size = 1
dtype = "float32"
model_name = "segnet"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "input0"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)


#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.
#
# To perform a precise measurement, we should repeat the measurement several
# times and use the average of results. In addition, we need to flush the cache
# for the weight tensors between repeated measurements. This can make the measured
# latency of one operator closer to its actual latency during end-to-end inference.

tuning_option = {
    'log_filename': log_file,
    'tuner': 'gridsearch',
    'early_stopping': 100,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=1000),
        runner=autotvm.LocalRunner(number=1, repeat=10,
                                   min_repeat_ms=0,
                                   enable_cpu_cache_flush=True),
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 tuner='xgb',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.op.get("nn.conv3d"),]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv3d"),))

    # run tuning tasks
    # tasks = [tasks[0]]
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # compile kernels with graph-level best records
    with autotvm.apply_history_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)
        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # save the model for project
        lib_path = 'auto_deploy_lib_x86.so'
        lib.export_library(lib_path)
        json_path = 'auto_deploy_graph_x86.json'
        params_path = 'auto_deploy_param_x86.params'
        with open(json_path, 'w') as of:
            of.write(graph)
        with open(params_path, 'wb') as of:
            of.write(relay.save_param_dict(params))

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended.
# One sample output is listed below.
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:  598.05/2497.63 GFLOPS | Progress: (252/252) | 1357.95 s Done.
#    [Task  2/12]  Current/Best:  522.63/2279.24 GFLOPS | Progress: (784/784) | 3989.60 s Done.
#    [Task  3/12]  Current/Best:  447.33/1927.69 GFLOPS | Progress: (784/784) | 3869.14 s Done.
#    [Task  4/12]  Current/Best:  481.11/1912.34 GFLOPS | Progress: (672/672) | 3274.25 s Done.
#    [Task  5/12]  Current/Best:  414.09/1598.45 GFLOPS | Progress: (672/672) | 2720.78 s Done.
#    [Task  6/12]  Current/Best:  508.96/2273.20 GFLOPS | Progress: (768/768) | 3718.75 s Done.
#    [Task  7/12]  Current/Best:  469.14/1955.79 GFLOPS | Progress: (576/576) | 2665.67 s Done.
#    [Task  8/12]  Current/Best:  230.91/1658.97 GFLOPS | Progress: (576/576) | 2435.01 s Done.
#    [Task  9/12]  Current/Best:  487.75/2295.19 GFLOPS | Progress: (648/648) | 3009.95 s Done.
#    [Task 10/12]  Current/Best:  182.33/1734.45 GFLOPS | Progress: (360/360) | 1755.06 s Done.
#    [Task 11/12]  Current/Best:  372.18/1745.15 GFLOPS | Progress: (360/360) | 1684.50 s Done.
#    [Task 12/12]  Current/Best:  215.34/2271.11 GFLOPS | Progress: (400/400) | 2128.74 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 3.16 ms (0.03 ms)
