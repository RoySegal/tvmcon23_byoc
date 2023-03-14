import os
from tvm import runtime

def save_lib(lib, graph, param):
    tvm_home = os.getenv('TVM_HOME')
    contrib_path = os.path.join(tvm_home, "src", "runtime", "contrib")
    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    lib.export_library(r"/projects/vbu_projects/users/roys/tvmcon23/out/model.so", fcompile=False, **kwargs)

    with open(r"/projects/vbu_projects/users/roys/tvmcon23/out/graph.json", "w") as f_graph_json:
        f_graph_json.write(graph)
    with open(r"/projects/vbu_projects/users/roys/tvmcon23/out/param.bin", "wb") as f_params:
        f_params.write(runtime.save_param_dict(param))

def load_lib():
    # Load model.so
    lib = runtime.module.load_module(r"/projects/vbu_projects/users/roys/tvmcon23/out/model.so")
    # Load graph.json
    with open(r"/projects/vbu_projects/users/roys/tvmcon23/out/graph.json", "r") as f_graph_json:
        graph = f_graph_json.read()
    # Load param.bin
    with open(r"/projects/vbu_projects/users/roys/tvmcon23/out/param.bin", "rb") as f_params:
        param = runtime.load_param_dict(f_params.read())
    return lib, graph, param