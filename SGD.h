#pragma once
#include "optimizer.h"


class SGD :public optimizer {
public:

	virtual void step();
};