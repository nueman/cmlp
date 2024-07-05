#include "linear.h"



// 一个thread计算一个out , out_size =  in @ weight
__global__
void linear_forwad_cuda(float* in, float* weights, float* bias, float* out,
	unsigned int  bs, unsigned int  in_len, unsigned int  out_len) {

	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

	if ((row < bs) && (col < out_len)) {

		unsigned int out_index = row * out_len + col;
		unsigned int in_index;
		unsigned int weights_index;

		float o = 0.0f; 
		for (unsigned int i = 0; i < in_len; i++) {
			in_index = row * in_len + i;
			weights_index = col * in_len + i;
			o = o + in[in_index] * weights[weights_index];
		}

		out[out_index] = o + bias[col];
	}

}






// same as pytorch
Linear::Linear(unsigned int _n_in, unsigned int _n_out , bool _use_bias ){
	this->in_len = _n_in;
	this->out_len = _n_out;
	auto s = std::vector<unsigned int>{  _n_out , _n_in };
	this->weights = new tensor(s);
	weights->linear_init(_n_in);


	auto bs = std::vector<unsigned int>{ 1,_n_out  };
	this->bias = new tensor(bs);


	this->use_bias = _use_bias;

	if (use_bias) {
		bias->linear_init(_n_in);
	}
		
	this->bias->requires_grad = use_bias;

}




void Linear::update(float lr) {

	this->weights->update(lr);
	if (use_bias) {
		this->bias->update(lr);
	}

}




Linear::~Linear() {
	delete this->weights;
	delete this->bias;
}


__global__
void linear_do_dw_cuda(float* in, float* out,
	unsigned int  w_col_len, unsigned int  w_row_len, unsigned int  bs,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col
	) {

	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int weights_index;
	//  out_len  row     in_len col  

	if ((row < w_row_len) && (col < w_col_len) ) {
		weights_index = row * w_col_len + col;

		for (unsigned int i = 0; i < bs; i++) {
			unsigned int in_index = i * w_col_len + col;
			auto out_g = in[in_index]; 

			unsigned int pre_index = i * w_row_len + row;
			out_g = out_g * pre_grad[pre_index];

			atomicAdd(&out[weights_index], out_g);
		}
	}
}


void c_linear_do_dw_cuda(dim3 n_blocks, dim3 n_threads,
	float* in, float* out,
	unsigned int  in_len, unsigned int  out_len, unsigned int  bs,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col
) {

	linear_do_dw_cuda << < n_blocks, n_threads >> > (in, out, in_len, out_len, bs , pre_grad , pg_row, pg_col);
	cudaDeviceSynchronize();
}



__global__
void linear_do_dx_cuda(
	float* weights, float* out,
	unsigned int  x_row_len, unsigned int  x_col_len, unsigned int  w_row_len, unsigned int  w_col_len,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col) {

	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int x_index;

	if ((row < x_row_len) && (col < x_col_len)) {
		x_index = row * x_col_len + col;

		for (unsigned int i = 0; i < w_row_len; i++) {

			unsigned int weights_index = i * w_col_len + col;
			auto out_g = weights[weights_index];

			unsigned int w_row = i;
			unsigned int pre_index =   row * pg_col + w_row;
			out_g = out_g * pre_grad[pre_index];

			atomicAdd(&out[x_index], out_g);
		}
	}
}


void c_linear_do_dx_cuda(dim3 n_blocks, dim3 n_threads,
	float* weights, float* out,
	unsigned int  x_row, unsigned int  x_col, unsigned int  w_row, unsigned int  w_col,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col

) {

	linear_do_dx_cuda << < n_blocks, n_threads >> > (weights, out, x_row, x_col, w_row , w_col, pre_grad, pg_row, pg_col);
	cudaDeviceSynchronize();
}





__global__
void linear_do_db_cuda(
    float* out,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col) {
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < pg_col) {
		for (unsigned int i = 0; i < pg_row; i++) {
			unsigned int pg_index = i * pg_col + index;
			auto out_g = pre_grad[pg_index];
			atomicAdd(&out[index], out_g);
		}
	}

}

void c_linear_do_db_cuda(unsigned int  n_blocks,
	float* out,
	float* pre_grad, unsigned int  pg_row, unsigned int  pg_col

) {

	linear_do_db_cuda << < n_blocks, block_size >> > ( out , pre_grad, pg_row, pg_col);
	cudaDeviceSynchronize();
}





tensor* Linear::forward(tensor* in, unsigned int bs) {


	check_shape(in, bs);

	auto out_s = std::vector<unsigned int>{ bs ,  out_len };
	tensor* out = new tensor(out_s);
	out->requires_grad = requires_grad;

	unsigned int n_block_rows = (bs + block_size - 1) / block_size;
	unsigned int n_block_cols = (out_len + block_size - 1) / block_size;

	dim3 n_threads(block_size, block_size);
	dim3 n_blocks(n_block_rows, n_block_cols);

	linear_forwad_cuda <<<n_blocks, n_threads>>> (in->value, weights->value, bias->value, out->value, bs, in_len, out_len);
	cudaDeviceSynchronize();


	out->parent_node.push_back(in);
	out->parent_node.push_back(weights);
	out->parent_node.push_back(bias);


	//this->weights->child_node.push_back(out);
	//in->child_node.push_back(out);
	//this->bias->child_node.push_back(out);


	if (this->requires_grad) {

		// w.shape
		unsigned int n_block_rows_b = (out_len  + block_size - 1) / block_size;
		unsigned int n_block_cols_b = (in_len + block_size - 1) / block_size;
		dim3 n_blocks_b(n_block_rows_b, n_block_cols_b);

		auto fn = std::bind(c_linear_do_dw_cuda, n_threads, n_blocks_b, in->value, weights->grad, in_len, out_len, bs,
		 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		weights->g_fn = fn;

	}
	

	if (in->requires_grad) {

		// x.shape
		unsigned int n_block_rows_b_i = (bs + block_size - 1) / block_size;
		unsigned int n_block_cols_b_i = (in_len + block_size - 1) / block_size;
		dim3 n_blocks_b_i(n_block_rows_b_i, n_block_cols_b_i);

		auto fn = std::bind(c_linear_do_dx_cuda, n_threads, n_blocks_b_i,
			weights->value, in->grad, bs, in_len, out_len ,in_len,
			 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

		in->g_fn = fn;
	
	}


	if (this->bias->requires_grad ) {
		// x.shape
	

		unsigned int  n_blocks = ( out_len + block_size - 1) / block_size;



		auto fn = std::bind(c_linear_do_db_cuda, n_blocks,
			bias->grad,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

		bias->g_fn = fn;

	}

	

	return out;
}



void Linear::zero_grad() {
	this->weights->zero_grad();
	if (use_bias) {
		this->bias->zero_grad();
	}

};



void read_weights(const char* name, Linear* l) {

	//auto len = l->in_len * l->out_len;

	using namespace std;
	string str;
	ifstream ifs;
	ifs.open(name, ios::in);

	while (getline(ifs, str)) {
		std::vector<std::string> ret = split(str, "b");
		int wl = 0;
		int bl = 0;

		auto sw = ret[0];
		std::vector<std::string> sbs;
		if (ret.size() == 2) {
			auto sb = ret[1];
			sbs = split(sb, " ");
		}


		std::vector<std::string> sws = split(sw, " ");

		for (int i = 0; i < sws.size(); i++) {
			if (sws[i] == "" || sws[i] == " " || sws[i] == "\n")
				continue;

			l->weights->value[wl] = std::stof(sws[i]);
			wl++;
		}

		for (int i = 0; i < sbs.size(); i++) {
			if (sbs[i] == "" || sbs[i] == " " || sbs[i] == "\n")
				continue;
			l->bias->value[bl] = std::stof(sbs[i]);
			bl++;
		}


	}


}


/*



*/