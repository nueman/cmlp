#include "relu.h"


__global__
void relu_forward_cuda(float* in, float* out, unsigned int out_size) {
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < out_size) {
		out[index] = fmaxf(0.0f, in[index]);
	}
}



__global__
void relu_do_cuda(float* in, float* out, unsigned int out_size,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col) {

	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < out_size) {
		auto g = in[index] > 00.f ? 1.0f : 0.0f;
		g = pre_grad[index] * g;

		out[index] = g;
	}

}


void c_relu_do_cuda(unsigned int  n_blocks , float* in, float* out, unsigned int out_size,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col) {
	relu_do_cuda << <n_blocks, block_size >> > (in, out, out_size , pre_grad, pg_row, pg_col);
	cudaDeviceSynchronize();
}



tensor* Relu::forward(tensor* in, unsigned int bs) {


	check_shape(in, bs);


	auto out_s = std::vector<unsigned int>{ bs ,  out_len };
	tensor*  out=new tensor(out_s);
	out->requires_grad = requires_grad;


	unsigned int  n_blocks = (bs * out_len + block_size - 1) / block_size;

	relu_forward_cuda <<<n_blocks, block_size>>> (in->value, out->value, bs * out_len);
	cudaDeviceSynchronize();

	out->parent_node.push_back(in);
	//in->child_node.push_back(out);


	auto fn = std::bind(c_relu_do_cuda, n_blocks, in->value, in->grad, bs * out_len,
		 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	in->g_fn = fn;

	
	return out;
}

