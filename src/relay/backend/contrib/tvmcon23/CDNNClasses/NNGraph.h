#pragma once

#include "NNDefines.h"
#include "NNOpBase.h"
#include "NNGraphInput.h"
#include <map>
typedef enum E_NN_GRAPH_DATATYPE
{
	E_NN_GRAPH_DATATYPE_FLOAT32,
	E_NN_GRAPH_DATATYPE_INT32,
	E_NN_GRAPH_DATATYPE_INT16,
	E_NN_GRAPH_DATATYPE_INT8,
	E_NN_GRAPH_DATATYPE_UINT8
};

class NNGraph
{
public:
	NNGraph() : mvGraph(),
		msName(""),
		ms32dataOpsNumber(0),
		ms32totalOpNumber(0),
		ms32OpNumber(0),
		mbIsQuantized(false),
		meGraphDataType(E_NN_GRAPH_DATATYPE_FLOAT32),
		mGraphInput()
	{}
	virtual ~NNGraph() {
		mapInputToOp.clear();
		mapOutputToOp.clear();
	}
	void setName(std::string val) { msName = val; }

	int addOp(NNOpBase * pOp)
	{
		mvGraph.push_back(pOp);
		std::pair<std::map<std::string, NNOpBase *>::iterator, bool> ret;
		ret = mNameToNNOp.insert(std::pair<std::string, NNOpBase *>(pOp->getOpName(), pOp));
		return !(ret.second) ? E_NNPARSER_ERROR_INSERT_EXISTING_NODE : E_NNPARSER_OK;
	}

	int addFinalOutputTensorName(std::string outputname)
	{
		if (mvGraphOutput.size() < NNPARSER_MAX_NETWORK_OUTPUTS)
		{
			mvGraphOutput.push_back(outputname);
			return E_NNPARSER_OK;
		}
		else
		{
			printf("The user can not run a network with more then %d output buffers to the CDNN \n", NNPARSER_MAX_NETWORK_OUTPUTS);
			return E_NNPARSER_ERROR_OPERATOR_MAX_OUTPUTS_REACHED;
		}
	}

	std::vector<std::string> getFinalOutputTensor() { return mvGraphOutput; }

	void setDataLayerNumber(int val) { ms32dataOpsNumber = val; }
	void setTotalOpNumber(int val) { ms32totalOpNumber = val; }
	void setOpNumber(int val) { ms32OpNumber = val; }
	void setGraphInput(NNGraphInput *graphInput) { mGraphInput = *graphInput; }
	void setIsQuantized(bool isQuantized) { mbIsQuantized = isQuantized; }
	void setDataType(E_NN_GRAPH_DATATYPE val) { meGraphDataType = val; }

	std::string getName() const { return msName; }
	NNOpBase * getOp(unsigned int i) { return mvGraph[i]; }
	size_t getGraphSize() { return mvGraph.size(); }
	int getDataLayerNumber() const { return ms32dataOpsNumber; }
	int getTotalOpNumber() const { return ms32totalOpNumber; }
	int getOpNumber() const { return ms32OpNumber; }
	bool getIsQuantized() { return mbIsQuantized; }
	char getBitWidth() { return mBitWidth; }
	NNOpBase* getNNOpFromName(std::string val) { return mNameToNNOp.find(val) != mNameToNNOp.end() ? mNameToNNOp[val] : NULL; }
	NNGraphInput  &getGraphInput() { return mGraphInput; }
	std::multimap<std::string, NNOpBase *> & getMapInputToOp() { return mapInputToOp; }
	std::multimap<std::string, NNOpBase *> & getMapOutputToOp() { return mapOutputToOp; }
	E_NN_GRAPH_DATATYPE getDataType() const { return meGraphDataType; }
	/* Replace the old-NNOp with the new-NNOp in the mvGraph */
	int replace(NNOpBase* pOldNNOp, NNOpBase* pNewNNOp);
	/* Erase the pNNOp in the mvGraph if it's exist */
	int erase(NNOpBase* pNNOp);

private:
	//vector holding the nodes of the neural network graph
	std::vector<NNOpBase *>	mvGraph;

	//multi map Input name to operation
	std::multimap<std::string, NNOpBase *> mapInputToOp;

	//multi map Output name to operation
	std::multimap<std::string, NNOpBase *> mapOutputToOp;

	/*! \brief vector holding the final output tensor names of the network */
	std::vector<std::string> mvGraphOutput;

	//map Node name to NNOp operation
	std::map<std::string, NNOpBase *> mNameToNNOp;

	//name of the neural network graph
	std::string	msName;
	/*! \brief is data  quantized . */
	bool mbIsQuantized;

	/*! \brief  data  bit width . */
	char mBitWidth;
	// Op that require binary data (e.g. weights)
	int ms32dataOpsNumber;

	// total Op number
	int ms32totalOpNumber;

	// Op number
	int ms32OpNumber;

	E_NN_GRAPH_DATATYPE meGraphDataType;

	// Graph input
	NNGraphInput mGraphInput;
};