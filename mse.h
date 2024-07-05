#pragma once
#include "module.h"


class Mse  {
public:


	unsigned int out_len;

	Mse() { out_len = 1; };
	tensor* forward(tensor* in, tensor* target,  unsigned int bs) ;


};


