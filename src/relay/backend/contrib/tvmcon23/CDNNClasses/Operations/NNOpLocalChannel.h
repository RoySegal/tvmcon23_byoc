#pragma once

#include "../NNOp.h"

/*! \brief Local channel op */
class NNOpLocalChannel :
	public NNOp
{
public:
	NNOpLocalChannel(E_NN_OPERATOR_TYPES eType) : NNOp(eType),
		mbBatchNormUseGlobalStats(true)
	{}
	virtual ~NNOpLocalChannel() {};
	void setWeight(NNTensor val) { mWeight = val; }
	void setBias(NNTensor val) { mBias = val; }
	void setBatchNormUseGlobalStats(bool val) { mbBatchNormUseGlobalStats = val; }
	bool isBatchNormUseGlobalStats() const { return mbBatchNormUseGlobalStats; }

	NNTensor & getWeight() { return mWeight; }
	NNTensor & getBias() { return mBias; }

private:
	/*! \brief  */
	bool mbBatchNormUseGlobalStats;

	/*! \brief Weight tensor */
	NNTensor mWeight;

	/*! \brief Bias tensor */
	NNTensor mBias;
};