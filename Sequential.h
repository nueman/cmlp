#pragma once


#include "module.h"



class Sequential : public Module {
public:
	std::vector<Module*> layers;

	std::vector<tensor*> l_outs;


	Sequential() {};

	virtual tensor* forward(tensor* in, unsigned int bs);
	void add_layer(Module* l);

	virtual void zero_grad();



	virtual void update(float lr) ;


	virtual std::string get_name() {
		return "Sequential";
	}


};

