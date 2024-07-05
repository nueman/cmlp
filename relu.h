#pragma once
#include "module.h"



class Relu : public Module {
public:

	unsigned int out_len;

	Relu(unsigned int _out_len) { out_len = _out_len; };
	virtual ~Relu() {};

	virtual std::string get_name() {
		return "Relu";
	}


	virtual tensor* forward(tensor* in,  unsigned int bs);
};