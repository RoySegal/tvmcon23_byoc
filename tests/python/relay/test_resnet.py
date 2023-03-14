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
import tvm
import onnx
from tvm.relay.backend.runtime import Runtime
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform




def get_onnx_model(model_path):
    model = onnx.load(model_path)
    input_name = [_input.name for _input in model.graph.input][0]
    input_shapes = tuple([[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input][0])
    return model, input_name, input_shapes


# get relay IR model
def get_relay_model(onnx_model):
    """
    from onnx model to relay IR model
    :param onnx_model: onnx model
    :return: mod - relay model,
    params - weights etc
    """
    # load onnx model
    mod, params = relay.frontend.from_onnx(onnx_model, dtype="float32")
    return mod, params


def test_resnet(model_path):
    if not tvm.get_global_func("relay.ext.NNOp", True):
        print("skip because NNOp codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 3, 224, 224)
    onnx_model, _, __ = get_onnx_model(model_path)
    ref_mod, params = get_relay_model(onnx_model)
    # print(ref_mod)
    mod = transform.AnnotateTarget(["NNOp"])(ref_mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    print(mod)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, param = relay.build(mod, target="llvm", params=params, runtime=Runtime("cpp"))



if __name__ == "__main__":
    print('works')
    model_path = '/projects/vbu_projects/users/achiar/CEVA_DEV/InV/dev_ai/NetworkExamples/ONNX/Opensource/resnet/resnet152/deploy.onnx'
    # model_path = '/projects/vbu_projects/users/achiar/CEVA_DEV/InV/dev_ai/NetworkExamples/ONNX/Opensource/resnet/resnet101_v2/deploy.onnx'
    # model_path = '/projects/vbu_projects/users/achiar/CEVA_DEV/InV/dev_ai/NetworkExamples/ONNX/Opensource/resnet/resnet50_v2/deploy.onnx'
    # model_path = '/projects/vbu_projects/users/achiar/CEVA_DEV/InV/dev_ai/NetworkExamples/ONNX/Opensource/resnet/resnet18/deploy.onnx'
    # model_path = '/projects/vbu_projects/users/achiar/CEVA_DEV/InV/dev_ai/NetworkExamples/ONNX/Opensource/resnet/resnet101_v1/deploy.onnx'
    # model_path = '/projects/vbu_projects/users/achiar/CEVA_DEV/InV/dev_ai/NetworkExamples/ONNX/Opensource/resnet/50_custom_node/deploy.onnx'
    test_resnet(model_path)
    print('worked')
