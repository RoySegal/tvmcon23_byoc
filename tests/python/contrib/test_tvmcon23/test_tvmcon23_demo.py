from onnx import load
from tvm.relay.frontend import from_onnx
from tvm.relay import transform, build
from tvm import transform as transform2
from tvm.relay.backend.runtime import Runtime
from tvm import cpu
from tvm.contrib import graph_executor
from numpy import asarray, reshape
from PIL.Image import open as open_image
from utils import save_lib, load_lib

block = load(r"/projects/vbu_projects/users/roys/tvmcon23/3node/deploy.onnx")
mod, params = from_onnx(block)

mod = transform.AnnotateTarget(["tvmcon23"])(mod)
mod = transform.MergeCompilerRegions()(mod)
mod = transform.PartitionGraph()(mod)

with transform2.PassContext(opt_level=3):
    graph, lib, param = build(mod, target="llvm", params=params, runtime=Runtime("cpp"))

# Save the outputs to files
save_lib(lib, graph, param)

# Load the runtime modules, graph json and params from file
lib, graph, param = load_lib()

image = open_image(r"/projects/vbu_projects/users/roys/tvmcon23/3node/random.jpg")
numpydata = asarray(image)
i_data = reshape(numpydata, [1, 3, 64, 64]).astype("float32")
map_inputs = {}
map_inputs["input.1"] = i_data

rt_mod = graph_executor.create(graph, lib, cpu())
for name, data in map_inputs.items():
    rt_mod.set_input(name, data)
rt_mod.set_input(**param)

rt_mod.run()