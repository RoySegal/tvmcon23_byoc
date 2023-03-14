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
from tvm.relay.backend.runtime import Runtime
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
import tensorflow as tf


def get_tf_model(model_path):
	with tf.gfile.GFile(model_path, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		return graph_def
	


def get_relay_model(model_path, input_name="input_node", input_shape=(1, 224, 224, 3), output_name="output_node"):
    graph_def = get_tf_model(model_path)
    with tvm.transform.PassContext(opt_level=3):
        mod, lib, params = tvm.relay.frontend.from_tensorflow(graph_def, layout="NHWC", shape={input_name: input_shape}, outputs=[output_name])
    return mod, params


def test_tf(model_path):
    if not tvm.get_global_func("relay.ext.NNOp", True):
        print("skip because NNOp codegen is not available")
        return

    ref_mod, params = get_relay_model(model_path, input_name="input_node", input_shape=(1, 224, 224, 3), output_name="output_node")
    print(ref_mod)
    mod = transform.AnnotateTarget(["NNOp"])(ref_mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, param = relay.build(mod, target="llvm", params=params, runtime=Runtime("cpp"))



if __name__ == "__main__":
    print('works')
    model_path = '/home/achiar/TVM_Dev/tvm/tests/python/relay/mobilenet/tf/v2_1.0_224/deploy.pb'
    test_tf(model_path)
    print('worked')
