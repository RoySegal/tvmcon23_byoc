#pragma once

#include <vector>
#include <cstdint>
#include "NNDefines.h"
#include "NNTensor.h"
#include "NNOpBase.h"

enum class E_NETWORK_TYPE : std::uint8_t {
	E_ONNX,
	E_TF,
	E_CAFFE,
	E_TF_LITE,
	E_ANDROID,
};

class NNOp : public NNOpBase
{
public:
	NNOp(E_NN_OPERATOR_TYPES eType) : NNOpBase(eType),
		meFusedActivation(E_FUSED_ACTIVATION::E_NNPARSER_FUSED_NONE)
	{}
	virtual ~NNOp() {}
	void setFusedActivation(E_FUSED_ACTIVATION val)
	{
		meFusedActivation = val;
	}
	int addInput(NNTensor *tensor)
	{
		if (mvInput.size() < NNPARSER_OPERATOR_MAX_INPUTS)
		{
			mvInput.push_back(*tensor);
			return E_NNPARSER_OK;
		}
		else
			return E_NNPARSER_ERROR_OPERATOR_MAX_INPUTS_REACHED;
	}
	int addOutput(NNTensor * tensor)
	{
		if (mvOutput.size() < NNPARSER_OPERATOR_MAX_OUTPUTS)
		{
			mvOutput.push_back(*tensor);
			return E_NNPARSER_OK;
		}
		else
			return E_NNPARSER_ERROR_OPERATOR_MAX_OUTPUTS_REACHED;
	}
	NNTensor * getOutput(int i)
	{
		if (i < mvOutput.size())
			return &mvOutput[i];
		else
			return NULL;
	}
	virtual NNTensor * getInput(int i)
	{
		if (i < mvInput.size())
			return &mvInput[i];
		else
			return NULL;
	}
	E_NETWORK_TYPE getNetworkType() {
		return meNetworkType;
	}
	int setNetworkType(E_NETWORK_TYPE val) {
		int s32status = 0;
		meNetworkType = val;
		return s32status;
	}
	unsigned int getInputSize() { return (unsigned int)mvInput.size(); }
	unsigned int getOutputSize() { return (unsigned int)mvOutput.size(); }
	E_FUSED_ACTIVATION getFusedActivation() { return (E_FUSED_ACTIVATION)meFusedActivation; }
	void setCustomParams(NNTensor val) { mCustomParams = val; }
	NNTensor & getCustomParams() { return mCustomParams; }

private:
	//List of input tensors of Type Tin
	std::vector<NNTensor>	mvInput;
	//List of output tensors of Type Tout
	std::vector<NNTensor> mvOutput;
	//ENUM of Fused Activation
	E_FUSED_ACTIVATION meFusedActivation;
	/* ! \brief CustomParams contains custom parameters taken from the customer's model
	and delivered as is into the customer's driver */
	NNTensor mCustomParams;
	E_NETWORK_TYPE meNetworkType;
};