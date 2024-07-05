#pragma once

#include "module.h"


class Linear : public Module {
public:

	tensor* weights=0;
	tensor* bias=0;
	unsigned int  in_len=0;
	unsigned int  out_len=0;
	bool use_bias = false;


	Linear(unsigned int _n_in, unsigned int _n_out,bool _use_bias = true);
	~Linear();


	virtual std::string get_name() {
		return "Linear";
	}


	virtual tensor* forward (tensor* in , unsigned int bs);


	virtual void zero_grad();


	virtual void update(float lr);


};


void read_weights(const char* name, Linear* l);