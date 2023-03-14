#pragma once

#include <vector>
#include <string>
#include "NNDefines.h"

/*! \brief Describes the type of layers supported by the current version of the NN
*/
typedef enum E_NN_OPERATOR_TYPES
{
	/*! \brief */
	E_NN_OPERATOR_NOTDEFINED, //reserved to 0 do not change
	/*! \brief */
	E_NN_OPERATOR_ABS,
	/*! \brief */
	E_NN_OPERATOR_ACOS,
	/*! \brief */
	E_NN_OPERATOR_ACCURACY,
	/*! \brief */
	E_NN_OPERATOR_ADD,
	/*! \brief */
	E_NN_OPERATOR_AND,
	/*! \brief */
	E_NN_OPERATOR_ARGMAX,
	/*! \brief */
	E_NN_OPERATOR_ARGMIN,
	/*! \brief */
	E_NN_OPERATOR_ASIN,
	/*! \brief */
	E_NN_OPERATOR_ATAN,
	/*! \brief */
	E_NN_OPERATOR_AVERAGEPOOL,
	/*! \brief */
	E_NN_OPERATOR_BATCHNORM,
	/*! \brief */
	E_NN_OPERATOR_BIDIRECTIONALRNN,
	/*! \brief */
	E_NN_OPERATOR_BILINEARRESIZE,
	/*! \brief */
	E_NN_OPERATOR_CAST,
	/*! \brief */
	E_NN_OPERATOR_CEIL,
	/*! \brief */
	E_NN_OPERATOR_CLIP,
	/*! \brief */
	E_NN_OPERATOR_CONCAT,
	/*! \brief */
	E_NN_OPERATOR_CONCATV2,
	/*! \brief */
	E_NN_OPERATOR_CONST,
	/*! \brief */
	E_NN_OPERATOR_CONV,
	/*! \brief */
	E_NN_OPERATOR_CONV_TRANSPOSE,
	/*! \brief */
	E_NN_OPERATOR_COS,
	/*! \brief */
	E_NN_OPERATOR_DIV,
	/*! \brief */
	E_NN_OPERATOR_DETECTIONOUTPUT,
	/*! \brief */
	E_NN_OPERATOR_DEPTHWISECONV2DNATIVE,
	/*! \brief */
	E_NN_OPERATOR_DROPOUT,
	/*! \brief */
	E_NN_OPERATOR_ELTWISE,
	/*! \brief */
	E_NN_OPERATOR_ELU,
	/*! \brief */
	E_NN_OPERATOR_EQUAL,
	/*! \brief */
	E_NN_OPERATOR_EXP,
	/*! \brief */
	E_NN_OPERATOR_FLATTEN,
	/*! \brief */
	E_NN_OPERATOR_FLOOR,
	/*! \brief */
	E_NN_OPERATOR_FUSEDBATCHNORM,
	/*! brief ONNX GRU type */
	E_NN_OPERATOR_GRU_PYTORCH,
	/*! brief */
	E_NN_OPERATOR_GATHER,
	/*! brief */
	E_NN_OPERATOR_GATHERV2,
	/*! brief */
	E_NN_OPERATOR_GEMM,
	/*! brief */
	E_NN_OPERATOR_GLOBALAVERAGEPOOL,
	/*! brief */
	E_NN_OPERATOR_GLOBALLPPOOL,
	/*! brief */
	E_NN_OPERATOR_GLOBALMAXPOOL,
	/*! brief */
	E_NN_OPERATOR_GREATER,
	/*! brief */
	E_NN_OPERATOR_HARDSIGMOID,
	/*! brief */
	E_NN_OPERATOR_HARDSWISH,
	/*! brief */
	E_NN_OPERATOR_HARDMAX,
	/*! brief */
	E_NN_OPERATOR_IDENTITY,
	/*! brief */
	E_NN_OPERATOR_INSTANCENORMALIZATION,
	/*! brief */
	E_NN_OPERATOR_LRN,
	/*! brief */
	E_NN_OPERATOR_LSTM,
	/*! brief */
	E_NN_OPERATOR_LSTM_BLOCK_FUSED_CELL,
	/*! brief */
	E_NN_OPERATOR_LSTM_BASIC_CELL,
	/*! brief */
	E_NN_OPERATOR_LEAKYRELU,
	/*! brief */
	E_NN_OPERATOR_LESS,
	/*! brief */
	E_NN_OPERATOR_LOG,
	/*! brief */
	E_NN_OPERATOR_LOGSOFTMAX,
	/*! brief */
	E_NN_OPERATOR_LPNORMALIZATION,
	/*! brief */
	E_NN_OPERATOR_LPPOOL,
	/*! brief */
	E_NN_OPERATOR_MATMUL,
	/*! brief */
	E_NN_OPERATOR_MAX,
	/*! brief */
	E_NN_OPERATOR_MAXPOOL,
	/*! brief */
	E_NN_OPERATOR_MAXROIPOOL,
	/*! brief */
	E_NN_OPERATOR_MEAN,
	/*! brief */
	E_NN_OPERATOR_MIN,
	/*! brief */
	E_NN_OPERATOR_MUL,
	/*! brief */
	E_NN_OPERATOR_MULTINOMIAL,
	/*! brief */
	E_NN_OPERATOR_NEG,
	/*! brief */
	E_NN_OPERATOR_NOT,
	/*! brief */
	E_NN_OPERATOR_OR,
	/*! brief */
	E_NN_OPERATOR_PRELU,
	/*! brief */
	E_NN_OPERATOR_PAD,
	/*! brief */
	E_NN_OPERATOR_POW,
	/*! brief */
	E_NN_OPERATOR_RNN,
	/*! brief */
	E_NN_OPERATOR_RANDOMNORMAL,
	/*! brief */
	E_NN_OPERATOR_RANDOMNORMALLIKE,
	/*! brief */
	E_NN_OPERATOR_RANDOMUNIFORM,
	/*! brief */
	E_NN_OPERATOR_RANDOMUNIFORMLIKE,
	/*! brief */
	E_NN_OPERATOR_RECIPROCAL,
	/*! brief */
	E_NN_OPERATOR_REDUCEL1,
	/*! brief */
	E_NN_OPERATOR_REDUCEL2,
	/*! brief */
	E_NN_OPERATOR_REDUCELOGSUM,
	/*! brief */
	E_NN_OPERATOR_REDUCELOGSUMEXP,
	/*! brief */
	E_NN_OPERATOR_REDUCEMEAN,
	/*! brief */
	E_NN_OPERATOR_REDUCEMAX,
	/*! brief */
	E_NN_OPERATOR_REDUCEMIN,
	/*! brief */
	E_NN_OPERATOR_REDUCEPROD,
	/*! brief */
	E_NN_OPERATOR_REDUCESUM,
	/*! brief */
	E_NN_OPERATOR_REDUCESUMSQUARE,
	/*! brief */
	E_NN_OPERATOR_RELU,
	/*! brief */
	E_NN_OPERATOR_RELU6,
  /*! brief */
	E_NN_OPERATOR_RESHAPE,
	/*! brief */
	E_NN_OPERATOR_SELU,
	/*! brief */
	E_NN_OPERATOR_SHAPE,
	/*! brief */
	E_NN_OPERATOR_SIGMOID,
	/*! brief */
	E_NN_OPERATOR_SIN,
	/*! brief */
	E_NN_OPERATOR_SIZE,
	/*! brief */
	E_NN_OPERATOR_SLICE,
	/*! brief */
	E_NN_OPERATOR_SOFTMAX,
	/*! brief */
	E_NN_OPERATOR_SOFTMAXWITHLOSS,
	/*! brief */
	E_NN_OPERATOR_SOFTPLUS,
	/*! brief */
	E_NN_OPERATOR_SOFTSIGN,
	/*! brief */
	E_NN_OPERATOR_SPACETODEPTH,
	/*! brief */
	E_NN_OPERATOR_SPLIT,
	/*! brief */
	E_NN_OPERATOR_SQRT,
	/*! brief */
	E_NN_OPERATOR_SQUEEZE,
	/*! brief */
	E_NN_OPERATOR_SUB,
	/*! brief */
	E_NN_OPERATOR_SUM,
	/*! brief */
	E_NN_OPERATOR_TAN,
	/*! brief */
	E_NN_OPERATOR_TANH,
	/*! brief */
	E_NN_OPERATOR_TILE,
	/*! brief */
	E_NN_OPERATOR_TOPK,
	/*! brief */
	E_NN_OPERATOR_TRANSPOSE,
	/*! brief */
	E_NN_OPERATOR_UNSQUEEZE,
	/*! brief */
	E_NN_OPERATOR_UPSAMPLE,
	/*! brief */
	E_NN_OPERATOR_XOR,
	/*! brief */
	E_NN_OPERATOR_ATEN,
	/*! brief */
	E_NN_OPERATOR_AFFINE,
	/*! brief */
	E_NN_OPERATOR_CONSTANTFILL,
	/*! brief */
	E_NN_OPERATOR_CROP,
	/*! brief tensorflow GRU type */
	E_NN_OPERATOR_GRU_TF,
	/*! brief */
	E_NN_OPERATOR_GIVENTENSORFILL,
	/*! brief */
	E_NN_OPERATOR_IF,
	/*! brief */
	E_NN_OPERATOR_IMAGESCALER,
	/*! brief */
	E_NN_OPERATOR_LOOP,
	/*! brief */
	E_NN_OPERATOR_LOOPINDEXTENSOR,
	/*! brief */
	E_NN_OPERATOR_MEANVARIANCENORMALIZATION,
	/*! brief */
	E_NN_OPERATOR_PARAMETRICSOFTPLUS,
	/*! brief */
	E_NN_OPERATOR_SCALE,
	/*! brief */
	E_NN_OPERATOR_SCALEDTANH,
	/*! brief */
	E_NN_OPERATOR_THRESHOLDEDRELU,
	/*! \brief */
	E_NN_OPERATOR_INNERPRODUCT,
	/*! \brief */
	E_NN_OPERATOR_INPUT,
	/*! \brief */
	E_NN_OPERATOR_NORMALIZE,
	/*! \brief */
	E_NN_OPERATOR_POOL,
	/*! \brief */
	E_NN_OPERATOR_PERMUTE,
	/*! \brief */
	E_NN_OPERATOR_PROPOSAL,
	/*! \brief */
	E_NN_OPERATOR_PRIORBOX,
	/*! \brief */
	E_NN_OPERATOR_PRIORBOX_TF,
	/*! \brief */
	E_NN_OPERATOR_ROI_POOLING,
	/*! \brief */
	E_NN_OPERATOR_PS_ROI_POOLING,
	/*! \brief */
	E_NN_OPERATOR_ROI_ALIGN,
	/*! \brief */
	E_NN_OPERATOR_ROI_ALIGN_TF,
	/*! \brief */
	E_NN_OPERATOR_CAROI_POOLING,
	/*! \brief */
	E_NN_OPERATOR_POWER,
	/*! \brief */
	E_NN_OPERATOR_REORG,
	/*! \brief */
	E_NN_OPERATOR_INTERPOLATION,
	/*! \brief */
	E_NN_OPERATOR_SHUFFLE_CHANNEL,
	/*! \brief */
	E_NN_OPERATOR_SPP,
	/*! brief */
	E_NN_OPERATOR_PACK,
	/*! brief */
	E_NN_OPERATOR_STRIDED_SLICE,
	/*! brief */
	E_NN_OPERATOR_CUSTOM,
	/*! \brief */
	E_NN_OPERATOR_GREEDY_DECODER,
	/*! \brief */
	E_NN_OPERATOR_LSTM_CONV_CELL,
	/*! \brief */
	E_NN_OPERATOR_SIGN,
	/*! \brief */
	E_NN_OPERATOR_PROD,
	/*! \brief */
	E_NN_OPERATOR_DEPTHTOSPACE,
	/*! \brief */
	E_NN_OPERATOR_EXPAND_DIMS,
	/*! brief */
	E_NN_OPERATOR_SWISH,
	/*! brief */
	E_NN_OPERATOR_MATMULV2,
	/*! brief */
	E_NN_OPERATOR_UNPACK,
	/*! brief */
	E_NN_OPERATOR_FAKEQUANTWITHMINMAXVARS,
	/*! brief */
	E_NN_OPERATOR_MISH,
	/*! brief */
	E_NN_OPERATOR_ATTENTION,
	/*! brief */
	E_NN_OPERATOR_LAYERNORM,
	/*! brief */
	E_NN_OPERATOR_EINSUM,
	/*! \brief */
	E_NN_OPERATOR_NOTSUPPORTED
};

// Op is the node of the graph
// Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a subsequent node.
class NNOpBase
{
public:
    NNOpBase() : meOperatorType(E_NN_OPERATOR_NOTDEFINED),
                 msOpName(""),
                 msType("")
    {}
	NNOpBase(E_NN_OPERATOR_TYPES eType) :
		ms32VendorId(0),
		ms32LibraryId(0),
		ms32NodeId(0),
		meOperatorType(eType)
	{}
	virtual ~NNOpBase() {}
	void setOpType(E_NN_OPERATOR_TYPES val) { meOperatorType = val; }
	void setOpName(std::string val) { msOpName = val; }
	void setStype(std::string val) { msType = val; }
	void setNodeId(unsigned int val) { ms32NodeId = val; }
	void setLibraryId(unsigned int val) { ms32LibraryId = val; }
	void setVendorId(unsigned int val) { ms32VendorId = val; }
	std::string				getOpName() const { return msOpName; }
	E_NN_OPERATOR_TYPES		getOpType() const { return meOperatorType; }
	std::string				getStype() const { return msType; }
	unsigned int			getVendorId() const { return ms32VendorId; }
	unsigned int			getLibraryId() const { return ms32LibraryId; }
	unsigned int			getNodeId() const { return ms32NodeId; }
private:
	//Vendor ID
	unsigned int				ms32VendorId;
	//Library ID
	unsigned int				ms32LibraryId;
	//Node ID
	unsigned int				ms32NodeId;
	//name of the operation node
	std::string					msOpName;
	//The enumaration of the operator to invoke.
	E_NN_OPERATOR_TYPES			meOperatorType;
	//The symbolic identifier of the operator to invoke.
	std::string					msType;
};