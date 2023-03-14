#include "../NNOp.h"

class NNOpInput :
	public NNOp
{
public:
	NNOpInput() : NNOp(E_NN_OPERATOR_INPUT),
		ms32NumOfInputs(0),
		ms32NumOfChannels(0),
		ms32Width(0),
		ms32Height(0)
	{}
	virtual ~NNOpInput(){}
	void setNumOfInputs(int val){ ms32NumOfInputs = val; }
	void setNumOfChannels(int val){ ms32NumOfChannels = val; }
	void setWidth(int val){ ms32Width = val; }
	void setHeight(int val){ ms32Height = val; }
	int getNumOfInputs() const { return ms32NumOfInputs; }
	int getNumOfChannels() const { return ms32NumOfChannels; }
	int getWidth() const { return ms32Width; }
	int getHeight() const { return ms32Height; }
private:
	int		ms32NumOfInputs;
	int		ms32NumOfChannels; // num_outputs in prototext
	int		ms32Width;
	int		ms32Height;
};