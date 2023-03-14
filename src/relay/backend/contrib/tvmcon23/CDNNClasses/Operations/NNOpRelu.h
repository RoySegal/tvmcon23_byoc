#pragma once


#include "NNOpLocalChannel.h"

/*  */
class NNOpRelu :
	public NNOpLocalChannel
{
public:
	NNOpRelu() : NNOpLocalChannel(E_NN_OPERATOR_RELU),
		mf32NegativeSlope(0)
	{}

	void setNegativeSlope(float val) { mf32NegativeSlope = val; }
	float getNegativeSlope() const { return mf32NegativeSlope; }
private:
	/*! \brief specifies whether to leak the negative part by multiplying it with the slope value rather than setting it to 0. */
	float mf32NegativeSlope;
};
