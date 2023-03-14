#pragma once

#include "NNDefines.h"
#include "NNOpBase.h"
#include "NNTensor.h"

class NNGraphInput
{ 
public:
    NNGraphInput() :mpsInput(),
        ms32InputCount(0),
        ms32InputImageCount(0),
        ms32InputInfoCount(0),
		m32InputMarkersSize(0)
    {}
	virtual ~NNGraphInput() {}

	void setInput(unsigned int i, std::string val) { mpsInput[i] = val; }
	void setInputSize(int val) { ms32InputCount = val; }
	void setInputImageCount(int val) { ms32InputImageCount = val; }
	void setInputInfoCount(int val) { ms32InputInfoCount = val; }
	void setInputMarkerCount(int val) { m32InputMarkersSize = val; }
	void setInputDims(unsigned int input, unsigned int dim, int val) { ms32InputDims[input][dim] = val; }

	std::string getInput(unsigned int i) const { return mpsInput[i]; }
	int getInputSize() const { return ms32InputCount; }
	int getInputImageCount() const { return ms32InputImageCount; }
	int getInputInfoCount() const { return ms32InputInfoCount; }
	int getInputMarkerCount() const { return m32InputMarkersSize; }
	int getInputDims(unsigned int input, unsigned int dim) const { return ms32InputDims[input][dim]; }

	NNTensor mInTensor[NNPARSER_MAX_NETWORK_INPUTS];

private:
	// input names
	std::string mpsInput[NNPARSER_MAX_NETWORK_INPUTS];

	// Input counter
	int ms32InputCount;

	// Input image counter
	int ms32InputImageCount;

	// Input information counter
	int ms32InputInfoCount;

	// Input clip markers counter
	int m32InputMarkersSize;

	// dimensions for each input
	int ms32InputDims[NNPARSER_MAX_NETWORK_INPUTS][NNPARSER_OPERATOR_MAX_DIMENSIONS];
};