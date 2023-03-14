

/*!
 * \file src/runtime/contrib/tvmcon23/tvmcon23_runtime.cc
 * \brief Qdata runtime for TVMCON23.
 */

#include "tvmcon23_runtime.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <fstream>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

TVMCON23Runtime::~TVMCON23Runtime() {
  // Release your devices resources
}

TVMCON23Runtime::TVMCON23Runtime(const std::string& symbol_name, const std::string& processed_subgraph, const Array<String> const_names)
    : const_names_(const_names),
    processed_subgraph_(processed_subgraph),
    symbol_name_(symbol_name) {
}

PackedFunc TVMCON23Runtime::GetFunction(const std::string& name,
                                    const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_symbol") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
  } else if (name == "get_const_vars") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_names_; });
  } else if (this->symbol_name_ == name) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      // Bind argument tensors to data entries.
      this->SetInputOutputBuffers(args);
      // Execute the subgraph.
      this->Run();
    });
  } else if ("__init_" + this->symbol_name_ == name) {
    // The function to initialize constant tensors.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1U);
      this->Init(args[0]);
      *rv = 0;
    });
  } else {
    return PackedFunc(nullptr);
  }
}

void TVMCON23Runtime::Init(const Array<NDArray>& consts) {
  std::cout << "Initializing sub-graph name: " << symbol_name_ << std::endl;
  // You SDK's API for the device's initialization goes here
}

void TVMCON23Runtime::Run() {
  std::cout << "Running sub-graph name: " << symbol_name_ << std::endl;
  // You SDK's API for the sub-graph execution goes here
}

void TVMCON23Runtime::SaveToBinary(dmlc::Stream* stream) {
  // Save the symbol
  stream->Write(symbol_name_);
  stream->Write(processed_subgraph_);
  // Save the required const names
  std::vector<std::string> consts;
  for (const auto& it : const_names_) {
    consts.push_back(it);
  }
  stream->Write(consts);
}

Module TVMCON23Runtime::LoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string symbol, processed_subgraph;
  std::vector<std::string> consts;

  // Load the symbol
  ICHECK(stream->Read(&symbol)) << "Loading symbol name failed";
  ICHECK(stream->Read(&processed_subgraph)) << "Loading symbol name failed";
  ICHECK(stream->Read(&consts)) << "Loading the const name list failed";
  Array<String> const_names;
  for (const auto& it : consts) {
    const_names.push_back(it);
  }

  auto n = tvm::runtime::make_object<TVMCON23Runtime>(symbol, processed_subgraph, const_names);
  return Module(n);
}
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tvmcon23").set_body_typed(TVMCON23Runtime::LoadFromBinary);

/*!
 * \brief Set up the input and output buffers by binding their DLTensor pointers to the
 * corresponding data entry.
 *
 * \param args The packed args.
 */
void TVMCON23Runtime::SetInputOutputBuffers(const TVMArgs& args) {
  for (size_t i = 0; i < static_cast<size_t>(args.size()); i++) {
    const DLTensor* arg;
    if (args[i].IsObjectRef<NDArray>()) {
      NDArray arr = args[i];
      arg = arr.operator->();
    } else {
      arg = args[i].operator DLTensor*();
    }

    data_entry_.push_back(arg);
  }
}

runtime::Module TVMCON23RuntimeCreate(String symbol_name, String processed_subgraph, const Array<String>& const_names) {
  auto n = make_object<TVMCON23Runtime>(symbol_name, processed_subgraph, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.TVMCON23RuntimeCreate").set_body_typed(TVMCON23RuntimeCreate);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
