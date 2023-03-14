#pragma once


#include "NNOpLocalChannel.h"

/*! \brief BatchNormalization Carries out batch normalization as described in the paper https://arxiv.org/abs/1502.03167. Depending on the mode it is being run, there are multiple cases for the number of outputs */
class NNOpBatchNormalization :
	public NNOpLocalChannel
{
public:
	NNOpBatchNormalization() : NNOpLocalChannel(E_NN_OPERATOR_BATCHNORM),
		f32Epsilon(1e-5f),
		f32Momentum(0.9f),
		f32Spatial(1)
	{}
	virtual ~NNOpBatchNormalization() {}

	void setEpsilon(float val) { f32Epsilon = val; }
	void setMomentum(float val) { f32Momentum = val; }
	void setSpatial(int val) { f32Spatial = val; }
	void setMean(NNTensor val) { mMean = val; }
	void setVariance(NNTensor val) { mVariance = val; }

	float getEpsilon() const { return f32Epsilon; }
	float getMomentum() const { return f32Momentum; }
	int getSpatial() const { return f32Spatial; }
	NNTensor & getMean() { return mMean; }
	NNTensor & getVariance() { return mVariance; }

private:
	/*! \brief The epsilon value to use to avoid division by zero, default is 1e-5f. */
	float f32Epsilon;

	/*! \brief Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum), default is 0.9f. */
	float f32Momentum;

	/*! \brief If true, compute the mean and variance across all spatial elements If false, compute the mean and variance across per feature.Default is 1. */
	int f32Spatial;

	/*! \brief Mean tensor */
	NNTensor mMean;

	/*! \brief Variance tensor */
	NNTensor mVariance;
};