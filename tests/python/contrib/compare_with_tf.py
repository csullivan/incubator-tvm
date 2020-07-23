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
# pylint: disable=import-self, invalid-name, unused-argument
"""
Tensorflow testcases
====================
This article is a test script to test tensorflow operator with Relay.
"""
from __future__ import print_function
import threading
import numpy as np
import pytest
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_functional_ops
from distutils.version import LooseVersion
import tvm
from tvm import te
from tvm import relay
import tvm.relay.testing.tf as tf_testing
from tvm.runtime.vm import VirtualMachine
from packaging import version as package_version

#######################################################################
# Generic run functions for TVM & tensorflow
# ------------------------------------------


def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

tf_dtypes = {
    'float32': tf.float32,
    'float16': tf.float16,
    'float64': tf.float64,
    'int32': tf.int32,
    'uint8' : tf.uint8,
    'int8': tf.int8,
    'int16': tf.int16,
    'uint16': tf.uint16,
    'int64': tf.int64,
}

def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy().tolist()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == 'Cons':
            tl = vmobj_to_list(o.fields[1])
            hd = vmobj_to_list(o.fields[0])
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == 'Nil':
            return []
        elif 'tensor_nil' in o.constructor.name_hint:
            return [0]
        elif 'tensor' in o.constructor.name_hint:
            return [o.fields[0].asnumpy()]
        else:
            raise RuntimeError("Unknown object type: %s" %
                               o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def run_tvm_graph(graph_def, input_data, input_node, num_output=1,
                  target='llvm', out_names=None, opt_level=3, mode='graph_runtime',
                  cuda_layout="NCHW", layout=None, disabled_pass=None, ignore_in_shape=False):
    """ Generic function to compile on relay and execute on tvm """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    if target == "cuda":
        layout = cuda_layout
    target_host = None
    if ignore_in_shape:
        shape_dict = None
    else:
        shape_dict = {e: i.shape if hasattr(i, "shape") else ()
                      for e, i in zip(input_node, input_data)}
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
                                                 shape=shape_dict,
                                                 outputs=out_names)
    ctx = tvm.context(target, 0)
    if mode == 'debug':
        ex = relay.create_executor(mode, mod=mod, ctx=tvm.cpu(), target="llvm")
        inputs = []
        for param in mod['main'].params:
            found = False
            for i, n in enumerate(input_node):
                if n == param.name_hint:
                    found = True
                    inputs.append(tvm.nd.array(input_data[i]))
                    break
            # Interpreter doesn't bind constants, so still need to find in params
            if not found:
                inputs.append(tvm.nd.array(params[param.name_hint]))
        result = ex.evaluate()(*inputs)
        return vmobj_to_list(result)
    elif mode == 'vm':
        with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass):
            vm_exec = relay.vm.compile(mod, target="llvm", params=params)
        vm = VirtualMachine(vm_exec)
        vm.init(tvm.cpu())
        inputs = {}
        for e, i in zip(input_node, input_data):
            inputs[e] = tvm.nd.array(i)
        result = vm.invoke("main", **inputs)
        return vmobj_to_list(result)
    else:
        with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass):
            graph, lib, params = relay.build(mod, target, target_host, params)
        from tvm.contrib import graph_runtime
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        for e, i in zip(input_node, input_data):
            m.set_input(e, tvm.nd.array(i))

        m.set_input(**params)
        # execute
        m.run()
        # get outputs
        assert out_names is None or num_output == len(out_names), (
            "out_names: {} num_output: {}".format(out_names, num_output))
        tvm_output_list = [m.get_output(i).asnumpy()
                           for i in range(num_output)]
        return tvm_output_list


def run_tf_graph(sess, input_data, input_node, output_node):
    """ Generic function to execute tensorflow """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    output_node = convert_to_list(output_node)

    tensor = [sess.graph.get_tensor_by_name(
        output_name) for output_name in output_node]

    input_dict = {e: input_data[i] for i, e in enumerate(input_node)}

    output_data = sess.run(tensor, input_dict)
    return output_data


def compare_tf_with_tvm(in_data, in_name, out_name, init_global_variables=False,
                        no_gpu=False, opt_level=3, mode='graph_runtime',
                        cuda_layout="NCHW"):
    """Generic function to generate and compare tensorflow and TVM output"""
    def name_without_num(name):
        return name.split(':')[0] if ":" in name else name

    out_name = convert_to_list(out_name)
    out_node = [name_without_num(name) for name in out_name]

    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    in_node = [name_without_num(name) for name in in_name]
    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        final_graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)

        #tf_output = run_tf_graph(sess, in_data, in_name, out_name)

        #for device in ["llvm", "cuda"]:
        for device in ["opencl"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue
            if no_gpu and device == 'cuda':
                continue

            tvm_output = run_tvm_graph(final_graph_def, in_data, in_node,
                                       target=device, out_names=out_name,
                                       num_output=len(out_name), opt_level=opt_level, mode=mode,
                                       cuda_layout=cuda_layout)


        sess.close()



#######################################################################
# Non Max Suppression
# -------------------
def _test_forward_nms_v3(bx_shape, score_shape, iou_threshold, score_threshold, out_size, dtype="float32"):
    boxes = np.random.uniform(0, 10, size=bx_shape).astype(dtype)
    scores = np.random.uniform(size=score_shape).astype(dtype)
    max_output_size = np.int32(out_size)
    tf.reset_default_graph()
    in_data_1 = tf.placeholder(dtype, boxes.shape, name="in_data_1")
    in_data_2 = tf.placeholder(dtype, scores.shape, name="in_data_2")
    in_data_3 = tf.placeholder(tf.int32, name="in_data_3")
    tf.image.non_max_suppression(boxes=in_data_1, scores=in_data_2, max_output_size=in_data_3,
                                 iou_threshold=iou_threshold, score_threshold=score_threshold, name="nms")
    compare_tf_with_tvm([boxes, scores, max_output_size], ['in_data_1:0', 'in_data_2:0', 'in_data_3:0'],
                        'nms/NonMaxSuppressionV3:0', mode='vm')
    compare_tf_with_tvm([boxes, scores, max_output_size], ['in_data_1:0', 'in_data_2:0', 'in_data_3:0'],
                        'nms/NonMaxSuppressionV3:0', mode='debug')

def _test_forward_nms_v4(bx_shape, score_shape, iou_threshold, score_threshold, out_size, dtype="float32"):
    boxes = np.random.uniform(0, 10, size=bx_shape).astype(dtype)
    scores = np.random.uniform(size=score_shape).astype(dtype)
    max_output_size = np.int32(out_size)
    tf.reset_default_graph()
    in_data_1 = tf.placeholder(dtype, boxes.shape, name="in_data_1")
    in_data_2 = tf.placeholder(dtype, scores.shape, name="in_data_2")
    in_data_3 = tf.placeholder(tf.int32, name="in_data_3")
    indices_padded, num_valid = tf.image.non_max_suppression_padded(boxes=in_data_1, scores=in_data_2, max_output_size=in_data_3,
                                 iou_threshold=iou_threshold, score_threshold=score_threshold, name="nms", pad_to_max_output_size=True)
    num_valid = tf.reshape(num_valid,shape=(-1,))
    indices_padded = tf.reshape(indices_padded, shape=(-1,))
    tf.slice(indices_padded, tf.constant([0]), num_valid, name="SlicedIndices")
    compare_tf_with_tvm([boxes, scores, max_output_size], ['in_data_1:0', 'in_data_2:0', 'in_data_3:0'],
                        ['nms/NonMaxSuppressionV4:1', "SlicedIndices:0"], mode='vm')
    compare_tf_with_tvm([boxes, scores, max_output_size], ['in_data_1:0', 'in_data_2:0', 'in_data_3:0'],
                        ['nms/NonMaxSuppressionV4:1',  "SlicedIndices:0"], mode='debug')

def test_forward_nms():
    """ NonMaxSuppressionV3,4 """
    for _test_forward_nms in [_test_forward_nms_v3, _test_forward_nms_v4]:
        _test_forward_nms((5, 4), (5,), 0.7, 0.5, 5)
        _test_forward_nms((20, 4), (20,), 0.5, 0.6, 10)
        _test_forward_nms((1000, 4), (1000,), 0.3, 0.7, 1000)
        _test_forward_nms((2000, 4), (2000,), 0.4, 0.6, 7)


