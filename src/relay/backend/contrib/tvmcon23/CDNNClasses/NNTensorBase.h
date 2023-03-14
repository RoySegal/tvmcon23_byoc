#pragma once

#include <string>

/*! \brief Defines the order of the data memory in the buffer class, where N is the number of samples, C is the number of channels, H is the height, and W is the width
*/
enum E_NN_MEMORY_PUBLIC_DATA_ORDER
{
	/*! \brief Sets the data dimension order: inputs, width, height, channels */
	E_NN_MEMORY_DATAORDER_NWHC,
	/*! \brief Sets the data dimension order: inputs, height, width, channels */
	E_NN_MEMORY_DATAORDER_NHWC,
	/*! \brief Sets the data dimension order: channels, width, height, inputs */
	E_NN_MEMORY_DATAORDER_CWHN,
	/*! \brief Sets the data dimension order: channels, height, width, inputs */
	E_NN_MEMORY_DATAORDER_CHWN,
	/*! \brief Sets the data dimension order: inputs, channels, width, height */
	E_NN_MEMORY_DATAORDER_NCWH,
	/*! \brief Sets the data dimension order: inputs, channels, height, width */
	E_NN_MEMORY_DATAORDER_NCHW,
	/*! \brief Sets the data dimension order: channels, inputs, width, height */
	E_NN_MEMORY_DATAORDER_CNWH,
	/*! \brief Sets the data dimension order: channels, inputs, height, width */
	E_NN_MEMORY_DATAORDER_CNHW,
	/*! \brief Sets the data dimension order: width, height, channels, inputs */
	E_NN_MEMORY_DATAORDER_WHCN,
	/*! \brief Sets the data dimension order: height, width, channels, inputs */
	E_NN_MEMORY_DATAORDER_HWCN,
	/*! \brief Sets the data dimension order: width, height, inputs, channels */
	E_NN_MEMORY_DATAORDER_WHNC,
	/*! \brief Sets the data dimension order: height, width, inputs, channels */
	E_NN_MEMORY_DATAORDER_HWNC,
	/*! \brief Gets the data order enum size */
	E_NN_MEMORY_DATAORDER_PUBLIC_SIZE
};

class NNTensorBase
{
public:
	NNTensorBase() : ms32Num(0),
		ms32Channels(0),
		ms32Height(0),
		ms32Width(0),
		mbvalid(false),
		meDataOrder(E_NN_MEMORY_DATAORDER_NCHW),
		msName(""),
		mf32ScaleFactor(1.0),
		mu32ZeroPoint(0),
		mbIsQuantized(false),
		mf32MaxVal(0),
		mf32MinVal(0),
		mu8BitWidth(16)
	{}
	virtual ~NNTensorBase(){}
	void setNum(int val) { ms32Num = val; }
	void setChannels(int val) { ms32Channels = val; }
	void setHeight(int val) { ms32Height = val; }
	void setWidth(int val) { ms32Width = val; }
	void setValid(bool val) { mbvalid = val; }
	void setDataOrder(E_NN_MEMORY_PUBLIC_DATA_ORDER val) { meDataOrder = val; }
	void setName(std::string val) { msName = val; }

	void setScaleFactor(float scaleFactor) { mf32ScaleFactor = scaleFactor; }
	void setZeroPoint(unsigned int zeroPoint) { mu32ZeroPoint = zeroPoint; }
	void setIsQuantized(bool isQuantized) { mbIsQuantized = isQuantized; }
	void setMinVal(float minVal) { mf32MinVal = minVal; }
	void setMaxVal(float maxVal) { mf32MaxVal = maxVal; }
	void setBitWidth(unsigned char bitWidth) { mu8BitWidth = bitWidth; }



	int getNum() const { return ms32Num; }
	int getChannels() const { return ms32Channels; }
	int getHeight() const { return ms32Height; }
	int getWidth() const { return ms32Width; }
	bool isValid() const { return mbvalid; }
	E_NN_MEMORY_PUBLIC_DATA_ORDER getDataOrder() const { return meDataOrder; }
	std::string getName() const { return msName; }
	float getScaleFactor() { return mf32ScaleFactor; }
	unsigned int  getZeroPoint() { return mu32ZeroPoint; }
	bool getIsQuantized() { return mbIsQuantized; }
	float getMinVal() { return mf32MinVal; }
	float getMaxVal() { return mf32MaxVal; }
	unsigned char getBitWidth() { return mu8BitWidth; }

private:
	/*! \brief valid marks this Tensor as valid data */
	bool			mbvalid;
	/*! \brief the order of the data  */
	E_NN_MEMORY_PUBLIC_DATA_ORDER meDataOrder;
	/*! \brief num is the batch size */
	int				ms32Num;
	/*! \brief number of channels */
	int				ms32Channels;
	/*! \brief height of the data. */
	int				ms32Height;
	/*! \brief width of the data. */
	int				ms32Width;
	/*! \brief scale factor of the quantized  data. */
	float			mf32ScaleFactor;
	/*! \brief zero point of the quantized  data. */
	unsigned int	mu32ZeroPoint;
	/*! \brief is data  quantized . */
	bool	        mbIsQuantized;
	/*! \brief minimal value of the quantized  data. */
	float			mf32MinVal;
	/*! \brief maximal value of the quantized  data. */
	float			mf32MaxVal;
	/*! \brief number of bits of each element in data. */
	unsigned char 	mu8BitWidth;
	/*! \brief unique name of the buffer */
	std::string			msName;
};
