#pragma once

#include <vector>
#include "NNTensorBase.h"

typedef enum E_NN_TENSOR_DIRECTION
{
	E_NN_TENSOR_IN,
	E_NN_TENSOR_OUT,
	E_NN_TENSOR_IN_OUT
};


typedef enum E_NN_TENSOR_DATATYPE
{
	E_NN_TENSOR_DATATYPE_FLOAT32,
	E_NN_TENSOR_DATATYPE_INT64,
	E_NN_TENSOR_DATATYPE_INT32,
	E_NN_TENSOR_DATATYPE_INT16,
	E_NN_TENSOR_DATATYPE_INT8,
	E_NN_TENSOR_DATATYPE_UINT8
};

typedef struct dataType_st
{
	enum E_NN_TENSOR_DATATYPE eDataType;
	unsigned int u32ByteWidth;
}dataType_st;


class NNTensor : public NNTensorBase
{
public:
	NNTensor() : NNTensorBase(),
		mpData(NULL),
		meDirection(E_NN_TENSOR_IN),
		meDataType(E_NN_TENSOR_DATATYPE_FLOAT32)
	{}
	virtual ~NNTensor(){}
	void setData(void* val) { mpData = val; }
	void setDirection(E_NN_TENSOR_DIRECTION val) { meDirection = val; }
	void setDataType(E_NN_TENSOR_DATATYPE val) { meDataType = val; }

	void* getData() const { return mpData; }
	E_NN_TENSOR_DIRECTION getDirection() const { return meDirection; }
	E_NN_TENSOR_DATATYPE getDataType() const { return meDataType; }

private:
	/*! \brief raw data of the tensor*/
	void* mpData;

	E_NN_TENSOR_DIRECTION meDirection;

	E_NN_TENSOR_DATATYPE meDataType;
};