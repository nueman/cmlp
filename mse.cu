#include "mse.h"



__global__
void mse_forward_cuda(float* in, float* target, float* out, unsigned int out_size) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < out_size) {
		auto t = in[index] - target[index];
		auto l = powf(t, 2) / out_size;
		atomicAdd(out, l);
	}
	
}


__global__
void mse_do_cuda(float* in, float* target, float* out, unsigned int out_size) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < out_size) {
		auto g = 2 * (in[index] - target[index]);
		g = g / out_size;
		out[index] = g;
	}

}


void c_mse_do_cuda(unsigned int  n_blocks, float* in, float* target , float* out, unsigned int out_size,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col) {

	mse_do_cuda << <n_blocks, block_size >> > (in, target,out  ,out_size);
	cudaDeviceSynchronize();

}


tensor*  Mse::forward(tensor* in, tensor* target, unsigned int bs) {


	auto out_s = std::vector<unsigned int>{ out_len ,  out_len };
	tensor* out = new tensor(out_s);

	unsigned int  n_blocks = (bs * out_len + block_size - 1) / block_size;

	mse_forward_cuda <<<n_blocks, block_size >>> (in->value, target->value, out->value, bs * out_len);
	cudaDeviceSynchronize();

	out->parent_node.push_back(in);
	out->parent_node.push_back(target);

	auto fn = std::bind(c_mse_do_cuda, n_blocks, in->value, target->value,  in->grad, bs * out_len,
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	in->g_fn = fn;


	out->grad[0] = 1.0f;

	return out;


}


