#pragma once

#include "util.h"
#include "tensor.h"





class Module {
public:
	virtual tensor* forward(tensor* in, unsigned int bs) { return 0; };

	bool requires_grad = true;
	 

	void check_shape(tensor* in, unsigned int bs) {
		if (bs != in->shape[0]) {
			std::cout << " bs != in.batch!!   " << std::endl;
			exit(0);
		}
	}


	virtual void zero_grad() {};

	virtual void update(float lr) {};

	virtual std::string get_name() {
		return "Module";
	}

};

