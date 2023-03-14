/*!
 * \file src/relay/backend/contrib/NNOp/relay_graph_parser.h
 * \brief Implementation of NNOp codegen APIs.
 */

//=============================================================================
//									Includes
//=============================================================================

#ifndef RELAYGRAPHPARSER_H
#define RELAYGRAPHPARSER_H

#pragma once
#include "../../utils.h"
#include "CDNNClasses/NNGraph.h"
#include "CDNNClasses/Operations/NNOpAll.h"

namespace tvm {
namespace relay {
namespace contrib {

//=============================================================================
//									Class API
//=============================================================================

/* return struct for VisitExpr func:
a way to pass a tensor from visitexpr func to parse func
In future may contain other fields to return*/
struct NNOpOutput {
  NNTensor tensor;
  std::vector<std::string> unique_names;
};

class RelayGraphParser : public backend::MemoizedExprTranslator<NNOpOutput> {
 public:
  explicit RelayGraphParser(const Function& func);

  ~RelayGraphParser();

  inline std::unique_ptr<NNGraph> GetGraph() { return std::move(nngraph_); }

  void CreateNNOpGraph();

 private:
  NNOpOutput VisitExprDefault_(const Object* op);

  NNOpOutput VisitExpr_(const VarNode* node);

  NNOpOutput VisitExpr_(const ConstantNode* cn);

  NNOpOutput VisitExpr_(const CallNode* call);

  NNOpOutput VisitExpr_(const TupleNode* node);

  NNOpOutput VisitExpr_(const TupleGetItemNode* node);

  void ParseNNOp_(const CallNode* call);

  void ParsePatterns_(const CallNode* call);

  void ParseNNOpInput_(const VarNode* node);

  void ParseNNOpConv_(const CallNode* call);

  void ParseConvAndBiasAdd_(const CallNode* call);

  void ParseNNOpRelu_(const CallNode* call);

  void ParseNNOpBatchNorm_(const CallNode* call);

  void AddInputTensor_(NNOp* pObj, NNOpOutput res, int idx = 0);

  void AddOutputTensor_(NNOp* pObj);

  void GetCommonParams_(const CallNode* call, NNOp* pObj);

  void SetInputsToOp_(const CallNode* call, NNOp* pObj);

  void AddOpToGraphRoutine_(NNOp* pObj, const CallNode* call,
                            bool is_common = true);

  NNTensor GetTensorData_(const ConstantNode* cn);

  void GetTensorData_(const ConstantNode* cn, NNTensor& tensor);

  std::string GetDtypeString_(const TensorTypeNode* ttype);

  std::unordered_set<std::string> SetTVMGraphInputNames_(
      tvm::Array<tvm::relay::Var> params);

  std::vector<const CallNode*> SetTVMGraphOutputs_(const Function& func);

  void UpdateGraphOutputs_(const CallNode* call);

  std::string GetCallNodeName_(const CallNode* call);

  const std::map<std::string,
                 std::pair<void (RelayGraphParser::*)(const CallNode* call),
                           E_NN_OPERATOR_TYPES>>
  SetOpMap_();

  void ParseConv_(const CallNode* call, NNOpConv* pConv);
  void SetConvParams_(const CallNode* call, NNOpConv* pConv);
  void SetConvParams_(const tvm::Array<tvm::PrimExpr>& strides,
                      const tvm::Array<tvm::PrimExpr>& padding, int groups,
                      const tvm::Array<tvm::PrimExpr>& dilation,
                      NNOpConv* pConv);
  inline std::string SetUniqueName_() {
    return "node_" + std::to_string(unique_name_counter_);
  }

  inline std::vector<const CallNode*> GetGraphOutputs_() {
    return graph_outputs_;
  }

  const Function& relay_graph_;
  std::unique_ptr<NNGraph> nngraph_;
  int unique_name_counter_;
  std::vector<const CallNode*> graph_outputs_;
  std::unordered_set<std::string> graph_input_names_;
  const std::map<std::string,
                 std::pair<void (RelayGraphParser::*)(const CallNode* call),
                           E_NN_OPERATOR_TYPES>>
      op_map_;

};  // end class RelayGraphParser

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif