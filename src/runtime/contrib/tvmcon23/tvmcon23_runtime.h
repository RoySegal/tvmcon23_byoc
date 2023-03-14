#ifndef TVM_RUNTIME_CONTRIB_TVMCON23_TVMCON23_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_TVMCON23_TVMCON23_RUNTIME_H_

#include <tvm/runtime/module.h>

#include <string>

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief TVMCON23 runtime for executing TVMCON23 models.
 * This class is a subclass of ModuleNode and provides a runtime for TVMCON23.
 */
class TVMCON23Runtime : public ModuleNode {
 public:
  /*!
   * \brief Constructor for TVMCON23 runtime.
   * \param symbol_name The name/symbol of the function.
   * \param qdata The QData of the function.
   * \param qdata_size The size of the QData.
   */
  TVMCON23Runtime(const std::string& symbol_name, const std::string& processed_subgraph, const Array<String> const_names);

  ~TVMCON23Runtime();

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override;

  /*!
   * \brief Get the type key of the TVMCON23 runtime.
   * \return The type key.
   */
  const char* type_key() const override { return "tvmcon23"; }

  /*!
   * \brief Save the TVMCON23 runtime to a binary stream.
   * \param stream The stream to save to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

  /*!
   * \brief Load the TVMCON23 runtime from a binary stream.
   * \param strm The stream to load from.
   * \return The loaded TVMCON23 runtime.
   */
  static Module LoadFromBinary(void* strm);

  /*!
   * \brief Initialize the TVMCON23 runtime.
   */
  void Init(const Array<NDArray>& consts);

  /*!
   * \brief Run the TVMCON23 runtime.
   */
  void Run();

 protected:
  /*!
   * \brief Set the input and output buffers for the TVMCON23 runtime.
   * \param args The input arguments.
   */
  void SetInputOutputBuffers(const TVMArgs& args);

 private:
  /*! \brief The name/symbol of the function. */
  std::string symbol_name_;
  /*! \brief The required constant names. */
  Array<String> const_names_;
  /*! \brief The processed subgraph. */
  std::string processed_subgraph_;
  /*! \brief Map the input name to entry id. */
  std::vector<uint32_t> input_var_eid_;
  /*! \brief Data of that entry. */
  std::vector<const DLTensor*> data_entry_;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TVMCON23_TVMCON23_RUNTIME_H_
