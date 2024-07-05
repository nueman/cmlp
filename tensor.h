#pragma once

#include "util.h"

// row
class tensor {
public:

	float* value = 0;
	std::vector<unsigned int> shape;


	unsigned int len = 0;
	unsigned int back_ward_count = 0;



	bool requires_grad = true ;
	bool print_backward = false;

	void linear_init(unsigned ol);

	virtual void update(float lr) ;
	void tensor::zero_grad();

	std::vector<tensor*> parent_node;
	std::vector<tensor*> child_node;


	float* grad = 0; // dL/dt 
	std::function<void(float* , unsigned int , unsigned int)> g_fn ;


	std::string op = "None";
	std::string name = "";

	tensor(unsigned int _len);
	tensor(std::vector<unsigned int>& _shape);
	void print();
	void print_grad();
	~tensor();
	void backward(tensor* pre) ;


private:
	tensor(const  tensor& t) {};
	tensor() {};
	void set_mem_zero();

};