#pragma once
#include "../NNOp.h"

/*! \brief NNOpConv  */
class NNOpConv :
	public NNOp
{
public:
	NNOpConv() : NNOp(E_NN_OPERATOR_CONV),
		ms32Group(1),
		mePaddingMode(E_PADDING_MODE::E_NNPARSER_PAD_MODE_DEFAULT),
		mvu32Pads(NNPARSER_MAX_PAD_DIM, 0),
		mvu32KernelShape(NNPARSER_OPERATOR_MAX_DIMENSIONS, 0),
		mvu32Strides(NNPARSER_OPERATOR_MAX_DIMENSIONS, 1),
		mvu32Dilations(NNPARSER_OPERATOR_MAX_DIMENSIONS, 1)
	{}

	NNOpConv(E_NN_OPERATOR_TYPES eType) : NNOp(eType),
		ms32Group(1),
		mePaddingMode(E_PADDING_MODE::E_NNPARSER_PAD_MODE_DEFAULT),
		mvu32Pads(NNPARSER_MAX_PAD_DIM, 0),
		mvu32KernelShape(NNPARSER_OPERATOR_MAX_DIMENSIONS, 0),
		mvu32Strides(NNPARSER_OPERATOR_MAX_DIMENSIONS, 1),
		mvu32Dilations(NNPARSER_OPERATOR_MAX_DIMENSIONS, 1)
	{}
	virtual ~NNOpConv() {};

	void setWeight(NNTensor val) { mWeight = val; }
	void setBias(NNTensor val) { mBias = val; }
	void setGroup(int val) { ms32Group = val; }
	void setPaddingMode(E_PADDING_MODE val) { mePaddingMode = val; }
	void setNumOutputChannels(int val){ ms32NumOutputChannels = val; }

	E_NNPARSER_ERROR setDilationsByIndex(unsigned int i, int val)
	{
		if (i > NNPARSER_OPERATOR_MAX_DIMENSIONS) return E_NNPARSER_ERROR_DIMENSIONS_MAX_LENGTH_REACHED;
		mvu32Dilations[i] = val;
		return E_NNPARSER_OK;
	}

	E_NNPARSER_ERROR setKernelShapeByIndex(unsigned int i, int val)
	{
		if (i > NNPARSER_OPERATOR_MAX_DIMENSIONS) return E_NNPARSER_ERROR_DIMENSIONS_MAX_LENGTH_REACHED;
		mvu32KernelShape[i] = val;
		return E_NNPARSER_OK;
	}
	E_NNPARSER_ERROR setStridesByIndex(unsigned int i, int val)
	{
		if (i > NNPARSER_OPERATOR_MAX_DIMENSIONS) return E_NNPARSER_ERROR_DIMENSIONS_MAX_LENGTH_REACHED;
		mvu32Strides[i] = val;
		return E_NNPARSER_OK;
	}
	E_NNPARSER_ERROR setPadsByIndex(unsigned int i, int val)
	{
		if (i > NNPARSER_MAX_PAD_DIM) return E_NNPARSER_ERROR_DIMENSIONS_MAX_LENGTH_REACHED;
		mvu32Pads[i] = val;
		return E_NNPARSER_OK;
	}

	NNTensor & getWeight() { return mWeight; }
	NNTensor & getBias() { return mBias; }
	E_PADDING_MODE getPaddingMode() const { return mePaddingMode; }
	int getGroup() const { return ms32Group; }
	int getNumOutputChannels() const { return ms32NumOutputChannels; }

	int getDilationByIndex(unsigned int i) const { return mvu32Dilations[i]; }
	int getKernelShapeByIndex(unsigned int i) const { return mvu32KernelShape[i]; }
	int getPadByIndex(unsigned int i) const { return mvu32Pads[i]; }
	int getStrideByIndex(unsigned int i) const { return mvu32Strides[i]; }
	unsigned int getDilationSize() const { return (unsigned int)mvu32Dilations.size(); }
	unsigned int getKernelShapeSize() const { return (unsigned int)mvu32KernelShape.size(); }
	unsigned int getPadSize() const { return (unsigned int)mvu32Pads.size(); }
	unsigned int getStrideSize() const { return (unsigned int)mvu32Strides.size(); }

private:
	/*! \brief Weight tensor */
	NNTensor mWeight;

	/*! \brief Bias tensor */
	NNTensor mBias;

	/*! \brief dilation value along each axis of the filter.
	If not present, the dilation defaults to 1 along each axis. */
	vector<unsigned int> mvu32Dilations;

	/*! \brief number of groups input channels and output channels are divided into, default is 1. */
	int ms32Group;

	/*! \brief the Padding mode of the convolution kernel. Default is 0. */
	E_PADDING_MODE mePaddingMode;

	/*! \brief The shape of the convolution kernel. If not present, should be inferred from input W. */
	vector<unsigned int> mvu32KernelShape;

	/*! \brief Padding for the beginning and ending along each axis,  */
	/* it can take any value greater than or equal to 0. The value represent
	the number of pixels added to the beginning and end part of the corresponding axis. */
	/*`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where
	xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
	the number of pixels added at the end of axis `i`. */
	/* This attribute cannot be used simultaneously with auto_pad attribute. If not present,
	the padding defaults to 0 along start and end of each axis. */
	vector<unsigned int> mvu32Pads;

	/*! \brief Stride along each axis. If not present, the stride defaults to 1 along each axis. */
	vector<unsigned int> mvu32Strides;

	/*! \brief Number of outputs in prototext*/
	int	ms32NumOutputChannels;
};
