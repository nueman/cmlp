#include "tensor.h"



#define DEBUG


void tensor::set_mem_zero() {


	for (unsigned int i = 0; i < this->len; i++) {
		this->value[i] = 0.0f;
		if (requires_grad) {
			this->grad[i] = 0.0f;
		}
	}

}




__global__
void update_cuda(float* value, float* grad, float lr  , unsigned int  row_len, unsigned int  col_len) {

	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;


	unsigned int index = row * row_len + col;

	auto g = -lr * grad[index];
	atomicAdd(&value[index], g);

}



 void tensor::update(float lr) {


	 unsigned int n_block_rows = (shape[0] + block_size - 1) / block_size;
	 unsigned int n_block_cols = (shape[1] + block_size - 1) / block_size;

	 dim3 n_threads(block_size, block_size);
	 dim3 n_blocks(n_block_rows, n_block_cols);

	 update_cuda<<<  n_blocks, n_threads  >>>(value,grad,lr, shape[0], shape[1]);
	 cudaDeviceSynchronize();

};




 void tensor::linear_init(unsigned ol) {

	 std::random_device rd;
	 std::mt19937 gen(rd());

	 auto b = sqrt(1.0 / ol);
	 std::uniform_real_distribution<float> dis(-b, b);

	 for (unsigned int i = 0; i < len; i++) {

		 auto v = dis(gen);
		 this->value[i] = v;
	 }

 };

void tensor::zero_grad() {
	if (requires_grad) {
		cudaMemset(this->grad, 0, this->len * sizeof(float));
		back_ward_count = 0;
	}
};







void tensor::backward(tensor* pre) {
	if (pre == 0) {
		std::cout << " no pre_grad! return!!!" << std::endl;
		return;
	}

	tensor* fpg = pre;

	if (g_fn) {

		if (back_ward_count > 0) {
			std::cout << " backward more than one !!!!" << std::endl;
			return;
		}

		back_ward_count++;
		
		float* pg = pre->grad;
		unsigned int pg_row = pre->shape[0];
		unsigned int pg_col = pre->shape[1];

		 g_fn(pg, pg_row, pg_col);

		 if (print_backward) {
			 std::cout << "-------------------------------------------------------------------" << std::endl;
			 std::cout << name + " pre_grad:" + pre->name + "" << std::endl;
			 printf(" pre_shape : %d ,%d  , value : \n", pg_row, pg_col);

			 std::cout << "" + name + ".grad:" << std::endl;
			 this->print_grad();
		 }


		 fpg = this;
	}


	for (int i = 0; i < this->parent_node.size(); i++) {
		parent_node[i]->backward(fpg);
	}
}

tensor::tensor(unsigned int _len ) {
	this->len = _len;
	cudaMallocManaged(&this->value, this->len * sizeof(float));
	this->shape.push_back(1);
	this->shape.push_back(_len);

	if (requires_grad) {
		cudaMallocManaged(&this->grad, this->len * sizeof(float));
	}

	set_mem_zero();
	
}


tensor::tensor(std::vector<unsigned int>& _shape) {
	this->shape.assign(_shape.begin(), _shape.end());
	this->len = 1;

	for (unsigned int i = 0; i < shape.size(); i++) {
		this->len = this->len * shape[i];
	}

	cudaMallocManaged(&this->value, this->len * sizeof(float));

	if (requires_grad) {
		cudaMallocManaged(&this->grad, this->len * sizeof(float));
	}

	set_mem_zero();
}


void tensor::print() {

	auto dim = this->shape.size();
	auto rl = this->shape[dim - 1];


	for (unsigned int i = 0; i < this->len; i++) {
		printf(" %f ", this->value[i]);
		if ((i + 1) % rl == 0) {
			printf(" \n ");
		}

	}


}


void tensor::print_grad() {

	if (!requires_grad) {
		printf("no grad\n");
		return;
	}


	auto dim = this->shape.size();
	auto rl = this->shape[dim - 1];


	for (unsigned int i = 0; i < this->len; i++) {
		printf(" %f ", this->grad[i]);
		if ((i + 1) % rl == 0) {
			printf(" \n ");
		}

	}

	printf(" \n ");

}

tensor :: ~tensor() {
	cudaFree(value);
	if (requires_grad) {
		cudaFree(grad);
	}

};