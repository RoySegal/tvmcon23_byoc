
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include "../../utils.h"

#include <fstream>

#include "relay_graph_parser.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module TVMCON23Compiler(const ObjectRef& ref) {
  auto func = Downcast<Function>(ref);
  std::string func_name(tvm::relay::backend::GetExtSymbol(func));
  std::cout << "Building subgraph: " << func_name << std::endl;
  Array<String> const_names = Array<String>();

  // Parse graph representation
  RelayGraphParser convertor(func);
  convertor.CreateNNOpGraph();
  const std::unique_ptr<NNGraph> nngraph = convertor.GetGraph();

  // Run generation, your SDK's API goes here!
  std::string processed_subgraph("Output Buffer");

  // Create a runtime module
  const auto* pf = runtime::Registry::Get("runtime.TVMCON23RuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find TVMCON23 runtime module to create";
  auto mod = (*pf)(func_name, processed_subgraph, const_names);
  return mod;
}
TVM_REGISTER_GLOBAL("relay.ext.tvmcon23").set_body_typed(TVMCON23Compiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
