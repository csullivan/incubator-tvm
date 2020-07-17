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
import os
import numpy as np
import mxnet.gluon as gluon
import tvm
from tvm import te
from tvm import relay
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
import onnx
import onnxruntime

import tvm.relay.testing.tf as tf_importer


from tvm.contrib.debugger import debug_runtime

def get_network(name, batch_size=None, is_gluon_model=True):
    if is_gluon_model:
        model = gluon.model_zoo.vision.get_model(name, pretrained=True)
    if "resnet50_v1" or "mobilenet1.0" in name:
        data_shape = (batch_size, 3, 224, 224)
    elif "inception" in name:
        data_shape = (batch_size, 3, 299, 299)
    else:
        raise ValueError("Unsupported network: " + name)

    return model, data_shape

def benchmark(tvm_mod, params, input_shape):
    target = 'llvm'
    ctx = tvm.cpu()
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(tvm_mod, target=target, params=params)
    test_data = np.random.normal(size=input_shape).astype('float32')
    m = debug_runtime.create(graph, lib, ctx)
    m.set_input('data', test_data)
    m.set_input(**params)
    m.run()
    return m.get_output(0)

def test_resnet50_ingestion():
    gluon_model, input_shape = get_network("resnet50_v1", batch_size=1)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
    benchmark(mod, params, input_shape)

def test_inceptionv3_ingestion():
    gluon_model, input_shape = get_network("inceptionv3", batch_size=1)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
    benchmark(mod, params, input_shape)

def test_mobilenetv1_ingestion():
    gluon_model, input_shape = get_network("mobilenet1.0", batch_size=1)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
    benchmark(mod, params, input_shape)

def test_vgg16_ingestion():
    gluon_model, input_shape = get_network("vgg16", batch_size=1)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
    benchmark(mod, params, input_shape)

def test_vgg16bn_ingestion():
    gluon_model, input_shape = get_network("vgg16_bn", batch_size=1)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data" : input_shape})
    benchmark(mod, params, input_shape)

def get_onnxruntime_output(model, inputs, dtype='float32'):
    import onnxruntime.backend
    rep = onnxruntime.backend.prepare(model, 'CPU')
    if isinstance(inputs, list) and len(inputs) > 1:
        ort_out = rep.run(inputs)
    else:
        x = inputs.astype(dtype)
        ort_out = rep.run(x)[0]
    return ort_out

def get_input_data_shape_dict(graph_def, input_data):
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict

def get_tvm_output(graph_def, input_data, output_shape=None, output_dtype='float32', opset=None):
    """ Generic function to execute and get tvm output"""
    target = 'llvm'
    ctx = tvm.cpu()

    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(graph_def, shape_dict, opset=opset)

    with tvm.transform.PassContext(opt_level=1):
        graph, lib, params = relay.build(mod,
                                         target,
                                         params=params)

    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_names):
            # Its possible for some onnx inputs to not be needed in the tvm
            # module, confirm its present before setting.
            try:
                m.set_input(input_names[i], tvm.nd.array(
                    input_data[i].astype(input_data[i].dtype)))
            except:
                continue
    else:
        m.set_input(input_names, tvm.nd.array(
            input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.asnumpy()

def test_mobilenetv3_ssdlite_ingestion(check = False):
    # TF pretrained model ssd_mobilenet_v3_small_coco
    # Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
    # Direct Tensorflow approach
    #graph_def = tf_importer.get_workload("/home/ubuntu/projects/tf-models-pretrained/ssd_mobilenet_v3_small_coco_2020_01_14/checkpoint/frozen_inference_graph.pb")
    graph_def = tf_importer.get_workload("/home/ubuntu/projects/mobilenetv3_ssdlite_tf1.15/frozen_inference_graph.pb")
    #tf.train.write_graph(graph_def, "./",name="mnv3-ssdlite-tf1.15export.pbtxt")
    graph_def = tf_importer.ProcessGraphDefParam(graph_def)
    #mod, params = relay.frontend.from_tensorflow(graph_def)
    if check:
        mod, params = relay.frontend.from_tensorflow(graph_def)
    else:
        mod, params = relay.frontend.from_tensorflow(graph_def, shape={"image_tensor": (1,320,320,3)})

    # # TF->ONNX approach
    # dtype = 'uint8'
    # data_shape = (1, 320, 320, 3)
    # x = np.random.uniform(size=data_shape).astype(np.uint8)
    # model = onnx.load_model("/home/ubuntu/projects/incubator-tvm/mobilenetv3_ssdlite.v11.onnx")
    # c2_out = get_onnxruntime_output(model, x, dtype)

    # tvm_outputs = get_tvm_output(model, x, output_shape=[1,2,3,4], output_dtype=[1,2,3,4])
    # # for target, ctx in ctx_list():
    # #     tvm_out = get_tvm_output(model, x, target, ctx, out_shape, dtype)
    # #     tvm.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)

    # Possible alternative: https://github.com/shaoshengsong/MobileNetV3-SSD


def test_deeplabv3_ingestion():
    graph_def = tf_importer.get_workload("/home/ubuntu/projects/tf-models-pretrained/deeplabv3_mnv2_pascal_train_aug/checkpoint/frozen_inference_graph.pb")
    tf.train.write_graph(graph_def, "./",name="deeplabv3_mobilenetv2.pbtxt")
    graph_def = tf_importer.ProcessGraphDefParam(graph_def)
    #mod, params = relay.frontend.from_tensorflow(graph_def)
    mod, params = relay.frontend.from_tensorflow(graph_def, shape={"ImageTensor": (1,320,320,3)})

if __name__ == "__main__":
    #test_resnet50_ingestion()
    #test_inceptionv3_ingestion()
    #test_mobilenetv1_ingestion()
    #test_vgg16_ingestion()
    test_mobilenetv3_ssdlite_ingestion()
    #test_deeplabv3_ingestion()
