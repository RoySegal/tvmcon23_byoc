#pragma once

#include "NNOpLocalChannel.h"

class NNOpScale :
	public NNOpLocalChannel
{
public:
	NNOpScale() : NNOpLocalChannel(E_NN_OPERATOR_SCALE),
		ms32Axis(1),
		mvs32Offsets(NNPARSER_OPERATOR_MAX_DIMENSIONS, 0)
	{}
	virtual ~NNOpScale(){};

	E_NNPARSER_ERROR setOffsetByIndex(unsigned int i, int val)
	{
		mvs32Offsets[i] = val;
		return E_NNPARSER_OK;
	}

	void setAxis(int val) { ms32Axis = val; }

	int getAxis() const { return ms32Axis; }
	unsigned int getOffset(int i) { return mvs32Offsets[i]; }
	unsigned int getOffsetSize() { return mvs32Offsets.size(); }
private:
	int				ms32Axis;
	vector<int>		mvs32Offsets;
};