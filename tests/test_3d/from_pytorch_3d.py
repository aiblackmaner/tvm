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
Compile PyTorch Models
======================
**Author**: `Alex Wong <https://github.com/alexwong/>`_

This article is an introductory tutorial to deploy PyTorch models with Relay.

For us to begin with, PyTorch should be installed.
TorchVision is also required since we will be using it as our model zoo.

A quick solution is to install via pip

.. code-block:: bash

    pip install torch==1.4.0
    pip install torchvision==0.5.0

or please refer to official site
https://pytorch.org/get-started/locally/

PyTorch versions should be backwards compatible but should be used
with the proper TorchVision version.

Currently, TVM supports PyTorch 1.4 and 1.3. Other versions may
be unstable.
"""

import tvm
from tvm import relay

import numpy as np
import time

from tvm.contrib.download import download_testdata
from tvm.contrib import util
from tvm.contrib.debugger import debug_runtime

# PyTorch imports
import torch
import torchvision

from test_3d.network import SegNet

######################################################################
# Load a pretrained PyTorch model
# -------------------------------
model_params = torch.load('state_params.pth.tar')
with torch.no_grad():
    net = SegNet(1, 3)
    net.eval()
    net.load_state_dict(model_params['state_dict'])

input_shape = [1, 1, 48, 48, 48]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(net, input_data).eval()

img = np.random.uniform(-1, 1, size=input_shape).astype("float32")

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = 'input0'
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = 'llvm'
target_host = 'llvm'
ctx = tvm.context(str(target), 0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
print('start runtime......')
start_time = time.time()
from tvm.contrib import graph_runtime
dtype = 'float32'
tvm_model = graph_runtime.GraphModule(lib["default"](ctx))
tvm_model.set_input(input_name, tvm.nd.array(img.astype(dtype)))
tvm_model.run()
tvm_output = tvm_model.get_output(0)
print('cost time:', time.time()-start_time)
print('runtime done!')

###########save model####################
print('start saving lib.......')
lib_path = 'opt_deploy_lib_x86.so'
lib.export_library(lib_path)
print('save lib done!')
#####################################################################

