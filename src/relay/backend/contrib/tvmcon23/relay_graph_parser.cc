/*!
 * \file src/relay/backend/contrib/NNOp/relay_graph_parser.cc
 * \brief Implementation of NNOp codegen APIs.
 */

//=============================================================================
//									              Includes
//=============================================================================

#include "relay_graph_parser.h"

#include <tvm/ir/error.h>
#include <tvm/relay/attrs/nn.h>

namespace tvm {
namespace relay {
namespace contrib {

//=============================================================================
//									              Functions
//=============================================================================

RelayGraphParser::RelayGraphParser(const Function& func)
    : relay_graph_(func),
      unique_name_counter_(-1),
      nngraph_(std::make_unique<NNGraph>()),
      graph_input_names_(SetTVMGraphInputNames_(func->params)),
      graph_outputs_(SetTVMGraphOutputs_(func)),
      op_map_(SetOpMap_()) {
  ICHECK(func.get()) << "Error: invalid input Function :" << std::endl;
  ICHECK(relay_graph_.get()) << "Error: can't find relay graph :" << std::endl;
  ICHECK(!graph_input_names_.empty())
      << "Error: can't find graph inputs" << std::endl;
  ICHECK(!graph_outputs_.empty())
      << "Error: can't find graph outputs" << std::endl;
}

RelayGraphParser::~RelayGraphParser() {}

void RelayGraphParser::CreateNNOpGraph() {
  auto out = VisitExpr(relay_graph_->body);
  nngraph_->setOpNumber(nngraph_->getGraphSize());
  nngraph_->setTotalOpNumber(nngraph_->getGraphSize());
}

NNOpOutput RelayGraphParser::VisitExprDefault_(const Object* op) {
  LOG(FATAL) << "NNOp codegen doesn't support: " << op->GetTypeKey();
  return {};
}

NNOpOutput RelayGraphParser::VisitExpr_(const VarNode* node) {
  NNOpOutput output;
  /* check if graph input and add NNopInput*/
  if (graph_input_names_.find(node->name_hint()) != graph_input_names_.end()) {
    ParseNNOpInput_(node);
  }
  output.unique_names.push_back(SetUniqueName_());
  return output;
}

NNOpOutput RelayGraphParser::VisitExpr_(const ConstantNode* cn) {
  NNOpOutput output;
  output.tensor = GetTensorData_(cn);
  return output;
}

NNOpOutput RelayGraphParser::VisitExpr_(const CallNode* call) {
  NNOpOutput output;
  ParseNNOp_(call);
  output.unique_names.push_back(SetUniqueName_());
  return output;
}

NNOpOutput RelayGraphParser::VisitExpr_(const TupleNode* node) {
  NNOpOutput output;
  for (auto expr_node : node->fields) {
    auto res = VisitExpr(expr_node);
    output.unique_names.push_back(res.unique_names[0]);
  }
  return output;
}

NNOpOutput RelayGraphParser::VisitExpr_(const TupleGetItemNode* node) {
  NNOpOutput output;
  auto tuple = node->tuple;
  auto res = VisitExpr(tuple);
  output.unique_names.push_back(res.unique_names[0]);
  return output;
}

void RelayGraphParser::ParseNNOp_(const CallNode* call) {
  std::string op_name = GetCallNodeName_(call);
  std::cout << op_name << std::endl;
  const auto it = op_map_.find(op_name);
  ICHECK(it != op_map_.end()) << "Operator not found";
  ((this->*(it->second.first)))(call);
}

void RelayGraphParser::ParseNNOpInput_(const VarNode* node) {
  NNOpInput* pInput = new NNOpInput;
  pInput->setOpType(E_NN_OPERATOR_INPUT);
  pInput->setStype("Input");
  // need to get the dims of input as in helper.cpp line 301
  auto ishape = backend::GetShape(node->checked_type());
  int s32Dims[4] = {-1, -1, -1, -1};
  for (size_t j = 0; j < ishape.size(); ++j) {
    s32Dims[j] = (int)ishape[j];
  }
  if (s32Dims[0] == 0) {
    s32Dims[0] = 1;
  }
  pInput->setNumOfInputs(s32Dims[0]);
  pInput->setNumOfChannels(s32Dims[1]);
  pInput->setHeight(s32Dims[2]);
  pInput->setWidth(s32Dims[3]);

  unique_name_counter_++;
  pInput->setOpName(SetUniqueName_());
  AddOutputTensor_(pInput);
  nngraph_->addOp(pInput);
}

void RelayGraphParser::ParseNNOpConv_(const CallNode* call) {
  NNOpConv* pConv = new NNOpConv;
  ParseConv_(call, pConv);
  AddOpToGraphRoutine_(pConv, call, false);
  nngraph_->setDataLayerNumber(nngraph_->getDataLayerNumber() + 1);
}

void RelayGraphParser::ParseNNOpRelu_(const CallNode* call) {
  NNOpRelu* pRelu = new NNOpRelu;
  SetInputsToOp_(call, pRelu);
  AddOpToGraphRoutine_(pRelu, call);
}

void RelayGraphParser::AddInputTensor_(NNOp* pObj, NNOpOutput res, int idx) {
  NNTensor tensor;
  tensor = NNTensor();
  tensor.setDirection(E_NN_TENSOR_IN);
  tensor.setName(res.unique_names[idx]);
  pObj->addInput(&tensor);
}

void RelayGraphParser::AddOutputTensor_(NNOp* pObj) {
  NNTensor tensor;
  tensor = NNTensor();
  tensor.setDirection(E_NN_TENSOR_OUT);
  tensor.setName(SetUniqueName_());
  // tensor.setUniqueNum(unique_name_counter_);
  pObj->addOutput(&tensor);
}

void RelayGraphParser::GetCommonParams_(const CallNode* call, NNOp* pObj) {
  // get op Node name and set NNOp sType
  // since its generic opertor's name
  pObj->setStype(GetCallNodeName_(call));
  // use the primitive op_name to set the opType
  pObj->setOpType(op_map_.find(GetCallNodeName_(call))->second.second);
}

void RelayGraphParser::SetInputsToOp_(const CallNode* call, NNOp* pObj) {
  for (const auto& arg : call->args) {
    auto res = VisitExpr(arg);
    for (int i = 0; i < res.unique_names.size(); i++) {
      AddInputTensor_(pObj, res, i);
    }
  }
}

void RelayGraphParser::AddOpToGraphRoutine_(NNOp* pObj, const CallNode* call,
                                            bool is_common) {
  if (is_common) GetCommonParams_(call, pObj);
  unique_name_counter_++;
  pObj->setOpName(SetUniqueName_());
  UpdateGraphOutputs_(call);
  AddOutputTensor_(pObj);
  nngraph_->addOp(pObj);
}

NNTensor RelayGraphParser::GetTensorData_(const ConstantNode* cn) {
  NNTensor tensor;
  tensor = NNTensor();
  tensor.NNTensorBase::setName("");
  GetTensorData_(cn, tensor);
  return tensor;
}

void RelayGraphParser::GetTensorData_(const ConstantNode* cn,
                                      NNTensor& tensor) {
  switch (cn->data->ndim) {
    case 0:
      tensor.NNTensorBase::setNum(1);
      tensor.NNTensorBase::setChannels(1);
      tensor.NNTensorBase::setHeight(1);
      tensor.NNTensorBase::setWidth(1);
      tensor.NNTensorBase::setValid(true);
      tensor.NNTensorBase::setDataOrder(E_NN_MEMORY_DATAORDER_NCHW);
      break;
    case 1:
      tensor.NNTensorBase::setNum((int)cn->data->shape[0]);
      tensor.NNTensorBase::setChannels(1);
      tensor.NNTensorBase::setHeight(1);
      tensor.NNTensorBase::setWidth(1);
      tensor.NNTensorBase::setValid(true);
      tensor.NNTensorBase::setDataOrder(E_NN_MEMORY_DATAORDER_NCHW);
      break;
    case 2:
      tensor.NNTensorBase::setNum((int)cn->data->shape[0]);
      tensor.NNTensorBase::setChannels((int)cn->data->shape[1]);
      tensor.NNTensorBase::setHeight(1);
      tensor.NNTensorBase::setWidth(1);
      tensor.NNTensorBase::setValid(true);
      tensor.NNTensorBase::setDataOrder(E_NN_MEMORY_DATAORDER_NCHW);
      break;
    case 3:
      tensor.NNTensorBase::setNum(1);
      tensor.NNTensorBase::setChannels((int)cn->data->shape[0]);
      tensor.NNTensorBase::setHeight((int)cn->data->shape[1]);
      tensor.NNTensorBase::setWidth((int)cn->data->shape[2]);
      tensor.NNTensorBase::setValid(true);
      tensor.NNTensorBase::setDataOrder(E_NN_MEMORY_DATAORDER_NCHW);
      break;
    case 4:
      tensor.NNTensorBase::setNum((int)cn->data->shape[0]);
      tensor.NNTensorBase::setChannels((int)cn->data->shape[1]);
      tensor.NNTensorBase::setHeight((int)cn->data->shape[2]);
      tensor.NNTensorBase::setWidth((int)cn->data->shape[3]);
      tensor.NNTensorBase::setValid(true);
      tensor.NNTensorBase::setDataOrder(E_NN_MEMORY_DATAORDER_NCHW);
      break;
  }

  const auto* type_node = cn->checked_type().as<TensorTypeNode>();
  if (GetDtypeString_(type_node) == "float") {
    tensor.setData((float*)cn->data->data);
    tensor.setDataType(E_NN_TENSOR_DATATYPE_FLOAT32);
  } else if (GetDtypeString_(type_node) == "int") {
    tensor.setData((int32_t*)cn->data->data);
    tensor.setDataType(E_NN_TENSOR_DATATYPE_INT32);
  } else if (GetDtypeString_(type_node) == "int64_t") {
    tensor.setData((int64_t*)cn->data->data);
    tensor.setDataType(E_NN_TENSOR_DATATYPE_INT64);
  }
}

std::string RelayGraphParser::GetDtypeString_(const TensorTypeNode* ttype) {
  std::string dtype;
  if (runtime::TypeMatch(ttype->dtype, kDLFloat, 32)) {
    dtype = "float";
  } else if (runtime::TypeMatch(ttype->dtype, kDLFloat, 16)) {
    dtype = "half";
  } else if (runtime::TypeMatch(ttype->dtype, kDLBfloat, 16)) {
    dtype = "bfloat";
  } else if (runtime::TypeMatch(ttype->dtype, kDLInt, 32)) {
    dtype = "int";
  } else if (runtime::TypeMatch(ttype->dtype, kDLInt, 64)) {
    dtype = "int64_t";
  } else {
    LOG(FATAL) << "Unsupported dtype " << ttype->dtype;
  }
  return dtype;
}

std::unordered_set<std::string> RelayGraphParser::SetTVMGraphInputNames_(
    tvm::Array<tvm::relay::Var> params) {
  std::unordered_set<std::string> input_names;
  for (const tvm::relay::Var& param : params) {
    input_names.insert(param->name_hint());
  }
  return input_names;
}

std::vector<const CallNode*> RelayGraphParser::SetTVMGraphOutputs_(const Function& func) {
  std::vector<const CallNode*> graph_outputs_;
  auto body = func->body;
  if (const auto* tuple = body.as<TupleNode>()) {
    // The body is a tuple expression, so loop over the fields of the tuple
    for (auto output : tuple->fields) {
      if (const auto* call = output.as<CallNode>()) {
        graph_outputs_.push_back(call);
      }
    }
  } else if (const auto* call = body.as<CallNode>()) {
    graph_outputs_.push_back(call);
  }
  return graph_outputs_;
}

void RelayGraphParser::UpdateGraphOutputs_(const CallNode* call) {
  if (std::find(graph_outputs_.begin(), graph_outputs_.end(), call) !=
      graph_outputs_.end()) {
    nngraph_->addFinalOutputTensorName(SetUniqueName_());
  }
}

std::string RelayGraphParser::GetCallNodeName_(const CallNode* call) {
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();
  return GetRef<Op>(op_node)->name;
}

const std::map<std::string,
               std::pair<void (RelayGraphParser::*)(const CallNode* call), E_NN_OPERATOR_TYPES>>
RelayGraphParser::SetOpMap_() {
  static const std::map<
      std::string, std::pair<void (RelayGraphParser::*)(const CallNode* call), E_NN_OPERATOR_TYPES>>
      op_map = {
          {"nn.conv2d", {&RelayGraphParser::ParseNNOpConv_, E_NN_OPERATOR_CONV}},
          {"nn.relu", {&RelayGraphParser::ParseNNOpRelu_, E_NN_OPERATOR_RELU}},
      };
  return op_map;
}

void RelayGraphParser::ParseConv_(const CallNode* call, NNOpConv* pConv) {
  GetCommonParams_(call, pConv);
  for (const auto& arg1 : call->args) {
    auto res = VisitExpr(arg1);
    if (arg1->IsInstance<ConstantNode>()) {
      pConv->setWeight(res.tensor);

      // update kernel fields according to weights tensor
      pConv->setKernelShapeByIndex(0, res.tensor.getHeight());
      pConv->setKernelShapeByIndex(1, res.tensor.getWidth());
      pConv->setKernelShapeByIndex(2, res.tensor.getChannels());
      pConv->setKernelShapeByIndex(3, res.tensor.getNum());
      if (E_NN_OPERATOR_CONV_TRANSPOSE == pConv->getOpType())
        pConv->setNumOutputChannels(res.tensor.getChannels() *
                                    pConv->getGroup());
      else
        pConv->setNumOutputChannels(res.tensor.getNum());
    } else  // input is former layer
    {
      AddInputTensor_(pConv, res);
    }
  }
  SetConvParams_(call, pConv);
}
void RelayGraphParser::SetConvParams_(const CallNode* call, NNOpConv* pConv) {
  if (E_NN_OPERATOR_CONV_TRANSPOSE == pConv->getOpType()) {
    const auto* conv2d_attr = call->attrs.as<Conv2DTransposeAttrs>();
    SetConvParams_(conv2d_attr->strides, conv2d_attr->padding,
                   conv2d_attr->groups, conv2d_attr->dilation, pConv);
  } else {
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    SetConvParams_(conv2d_attr->strides, conv2d_attr->padding,
                   conv2d_attr->groups, conv2d_attr->dilation, pConv);
  }
}
void RelayGraphParser::SetConvParams_(const tvm::Array<tvm::PrimExpr>& strides,
                                      const tvm::Array<tvm::PrimExpr>& padding,
                                      int groups,
                                      const tvm::Array<tvm::PrimExpr>& dilation,
                                      NNOpConv* pConv) {
  // # strides
  for (int iter = 0; iter < strides.size(); iter++) {
    pConv->setStridesByIndex(iter, (int)strides[iter].as<IntImmNode>()->value);
  }
  // # padding
  for (int iter = 0; iter < padding.size(); iter++) {
    pConv->setPadsByIndex(iter, (int)padding[iter].as<IntImmNode>()->value);
  }
  // # groups
  pConv->setGroup(groups);
  // # dilation
  for (int iter = 0; iter < dilation.size(); iter++) {
    pConv->setDilationsByIndex(iter,
                               (int)dilation[iter].as<IntImmNode>()->value);
  }

}
}  // namespace contrib
}  // namespace relay
}  // namespace tvm