
#include "util.h"
#include "tensor.h"
#include "linear.h"
#include "relu.h"
#include "Sequential.h"
#include "mse.h"
#include "SGD.h"





void test_vector() {

}


void test_linear_forward() {

	Linear l(5, 5);
	float w = 0.1f;
	for (int i = 0; i < 5; i++) {
		
		for (int j = 0; j < 5; j++) {
			l.weights->value[i * 5 + j] = w;
		}
		w = w + 0.1f;
	}
	
	l.weights->print();


	tensor a(50);


	w = 1.0f;
	for (int i = 0; i < 10; i++) {

		for (int j = 0; j < 5; j++) {
			a.value[i * 5 + j] = w;
		}
		w = w + 1.0f;
	}


	auto out = l.forward(&a, 10);

	printf("---------------------------------\n");
	out->print();

}

/*

tensor([[ 0.1366],
		[ 0.9931],
		[-0.0197],
		[-1.0152],
		[-0.2174]], device='cuda:0', grad_fn=<MmBackward0>)

*/


/*
def write_w( l ,name , ist=False ):
	w = l.weight
	if(ist):
		w = w.T
	w2 = w.reshape(-1)
	b = l.bias
	with open("./" + name +".txt", "a") as file:
		for i in range(w2.__len__()):
			file.write(str(w2[i].item()))
			file.write(" ")
		if(b!=None):
			file.write(" b ")
			for i in range(b.view(-1).__len__()):
				file.write(str(b.view(-1)[i].item()))
				file.write(" ")
		file.write("\n")
*/
void test_load_file() {

	Linear l0(1, 100);  // in_shape out_shape
	Linear l2(100, 100);
	Linear l4(100, 100);
	Linear l6(100, 1,false);
	Relu relu1(100);   //  in_shape = out_shape 参数非必须
	Relu relu3(100);
	Relu relu5(100);

	l0.requires_grad = false;
	l2.requires_grad = false;
	l4.requires_grad = false;
	l6.requires_grad = false;
	relu1.requires_grad = false;
	relu3.requires_grad = false;
	relu5.requires_grad = false;



	read_weights("nl0T.txt", &l0);
	read_weights("nl2T.txt", &l2);
	read_weights("nl4T.txt", &l4);
	read_weights("nl6T.txt", &l6);


	auto shape = std::vector<unsigned int>{ 5,1 };

	tensor in(shape);

	in.requires_grad = false;

	in.value[0] = 0.0f;
	in.value[1] = 0.25f;
	in.value[2] = 0.5f;
	in.value[3] = 0.75f;
	in.value[4] = 1.0f;



	Sequential s;

	s.add_layer(&l0);
	s.add_layer(&relu1);
	s.add_layer(&l2);
	s.add_layer(&relu3);
	s.add_layer(&l4);
	s.add_layer(&relu5);
	s.add_layer(&l6);


	auto out = s.forward(&in,  5);

	for (int i = 0; i < 5; i++) {
		printf(" %f ", out->value[i]);
	}



}


/*

ae = torch.tensor( [[1.,1.,1.]  ,[2.,2.,2.] ,[3.,3.,3.] ,[1.,1.,1.] ,[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]  ] , requires_grad=True )
ae.retain_grad()
l3 = nn.Linear(3,3,False)
w3 = torch.tensor([[0.1,0.1,0.1] , [0.2,0.2,0.2] , [0.3,0.3,0.3]])
l3.weight=torch.nn.Parameter(w3)
oo = l3(ae)
oo.backward(torch.tensor([[1.,1.,1.] , [1.,2.,3.],[5.,5.,1.],[1.,1.,1.],[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]  ]))

*/

void test_linear_backward() {


	Linear l(3, 3);

	float w = 0.1f;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			l.weights->value[i * 3 + j] = w;
		}
		w = w + 0.1f;
	}

	auto s = std::vector<unsigned int>{ 7,3 };
	tensor a( s );
	//a.requires_grad = false;

	for (int i = 0; i < 21; i++) {
		if (i < 3) {
			a.value[i] = 1.0f;
		}
		else if (i<6 ){
			a.value[i] = 2.0f;
		}
		else if (i < 9) {
			a.value[i] = 3.0f;
		}
		else {
			a.value[i] = 1.0f;
		}

	}


	auto out = l.forward(&a, 7);


	printf("\n-------------out---------------------\n");
	out->print();
	printf("\n-------------l.weights.grad---------------------\n");


	auto pgs = std::vector<unsigned int>{ 7,3 };
	tensor pg(pgs);

	for (int i = 0; i < 21; i++) {
			pg.grad[i] = 1.0f;
	}


    pg.grad[4] = 2.0f;
	pg.grad[5] = 3.0f;
	pg.grad[6] = 5.0f;
	pg.grad[7] = 5.0f;

	out->backward(&pg);

}









/*

relu = nn.ReLU()
l1 = nn.Linear(3,3)
w1 = torch.tensor([[0.1,0.1,0.1] , [-0.2,-0.2,-0.2] , [0.3,0.3,0.3]])
l1.weight=torch.nn.Parameter(w1)
w1b=torch.tensor([0.,0.,0.])
l1.bias=torch.nn.Parameter(w1b)
l2 = nn.Linear(3,3)
w2 =  torch.tensor([[0.1,0.1,0.1] , [0.2,0.2,0.2] , [0.3,0.3,0.3]])
l2.weight=torch.nn.Parameter(w2)
w2b=torch.tensor([0.,0.,0.])
l2.bias=torch.nn.Parameter(w2b)
l3 = nn.Linear(3,1,False)
w3 = torch.tensor([[0.1, - 0.2, 0.3] ])
l3.weight=torch.nn.Parameter(w3)
mse = nn.MSELoss()
a = torch.tensor( [[1.,1.,1.],[2,2,2] ],requires_grad=True)
a.retain_grad()
e1 = l1(a)
e1.retain_grad()
e2 = relu(e1)
e2.retain_grad()
e3 = l2(e2)
e3.retain_grad()
e4 = l3(e3)
t = torch.tensor( [[0.08] ,[0.16]])
loss = mse(e4,t)
loss.retain_grad()
loss.backward()

*/
void test_s_backward() {

	Linear l1(3, 3);
	l1.weights->name = "l1_w";
	l1.bias->name = "l1_bias";

	Linear l2(3, 3);
	l2.weights->name = "l2_w";
	l2.bias->name = "l2_bias";

	Linear l3(3, 1,false);
	l3.weights->name = "l3_w";
	l3.bias->name = "l3_bias";

	Relu relu(3);


	float w = 0.1f;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			l1.weights->value[i * 3 + j] = w;
		}
		w = w + 0.1f;
	}

	l1.weights->value[3] = -0.2f;
	l1.weights->value[4] = -0.2f;
	l1.weights->value[5] = -0.2f;

	w = 0.1f;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			l2.weights->value[i * 3 + j] = w;
		}
		w = w + 0.1f;
	}


	l3.weights->value[0] =  0.1f;
	l3.weights->value[1] = -0.2f;
	l3.weights->value[2] =  0.3f;

	Sequential s;


	s.add_layer(&l1);
	s.add_layer(&relu);
	s.add_layer(&l2);
	s.add_layer(&l3);
	Mse mse;

	auto sa = std::vector<unsigned int>{ 2, 3 };
	tensor a(sa);

	a.name = "x";


	for (int i = 0; i < 6; i++) {
		if (i < 3) {
			a.value[i] = 1.0f;
		}
		else {
			a.value[i] = 2.0f;
		}
	}


	auto s_out = s.forward(&a,2);

	printf("\n=============out================\n");

	s_out->print();

	printf("\n============= loss ================\n");

	tensor target(2);

	target.value[0] = 0.08f;
	target.value[1] = 0.16f;
	target.name = "target_tensor";

	auto loss =  mse.forward(s_out, &target, 2);
	loss->print();


	printf("\n=============grad ================\n");

	auto pgs = std::vector<unsigned int>{ 1,1 };
	tensor pg(pgs);
	pg.grad[0] = 1.0f;
	loss->backward(&pg);



}



void test_load_file_g() {

	Linear l0(1, 100);
	Linear l2(100, 100);
	Linear l4(100, 100);
	Linear l6(100, 1 ,false);

	l0.weights->name = "l0_w";
	l0.bias->name = "l0_bias";

	l2.weights->name = "l2_w";
	l2.bias->name = "l2_bias";

	l4.weights->name = "l4_w";
	l4.bias->name = "l4_bias";

	l6.weights->name = "l6_w";
	l6.bias->name = "l6_bias";


	Relu relu1(100);
	Relu relu3(100);
	Relu relu5(100);


	read_weights("tsg0.txt", &l0);
	read_weights("tsg2.txt", &l2);
	read_weights("tsg4.txt", &l4);
	read_weights("tsg6.txt", &l6);


	auto shape = std::vector<unsigned int>{ 5,1 };

	tensor in(shape);

	in.requires_grad = false;
	in.value[0] = 0.0f;
	in.value[1] = 0.25f;
	in.value[2] = 0.5f;
	in.value[3] = 0.75f;
	in.value[4] = 1.0f;


	Sequential s;

	s.add_layer(&l0);
	s.add_layer(&relu1);
	s.add_layer(&l2);
	s.add_layer(&relu3);
	s.add_layer(&l4);
	s.add_layer(&relu5);
	s.add_layer(&l6);


	auto out = s.forward(&in, 5);

	auto st = std::vector<unsigned int>{ 5, 1 };
	tensor target(st);

	target.value[0] = 0.0f;
	target.value[1] = 1.0f;
	target.value[2] = 0.0f;
	target.value[3] = -1.0f;
	target.value[5] = 0.0f;
	target.name = "target_tensor";

	Mse mse;

	printf("\n============= out ================\n");
	out->print();

	printf("\n============= loss ================\n");
	auto loss = mse.forward(out, &target, 5);
	loss->print();

	printf("\n============= grad ================\n");
	auto pgs = std::vector<unsigned int>{ 1,1 };
	tensor pg(pgs);
	pg.grad[0] = 1.0f;
	loss->backward(&pg);



}



void get_sin_data(std::vector<tensor*>& x , std::vector<tensor*>& y ,unsigned int batch , unsigned int batch_size ) {

	using namespace std;
	uniform_real_distribution<float> dis(0, 1);

	std::random_device rd;
	std::mt19937 gen(rd());



	for (unsigned int i = 0; i < batch; i++) {
		auto shape = std::vector<unsigned int>{ batch_size,1 };
		tensor* in = new tensor(shape);
		in->requires_grad = false;
		tensor* out = new tensor(shape);


		for (unsigned int j = 0; j < batch_size; j++) {
			float a = dis(rd);
			float b = sin(a * 2 * 3.1415926);
			in->value[j] = a;
			out->value[j] = b;
		}
		x.push_back(in);
		y.push_back(out);
	
	}

}

void train() {

	std::vector<tensor*> x;
	std::vector<tensor*> y;


	int batch_size = 256;
	int batch = 128;
	get_sin_data(x, y, batch, batch_size);


	Linear l0(1, 100);
	Linear l2(100, 100);
	Linear l4(100, 100);
	Linear l6(100, 1, false);

	l0.weights->name = "l0_w";
	l0.bias->name = "l0_bias";
	l2.weights->name = "l2_w";
	l2.bias->name = "l2_bias";
	l4.weights->name = "l4_w";
	l4.bias->name = "l4_bias";
	l6.weights->name = "l6_w";
	l6.bias->name = "l6_bias";


	Relu relu1(100);
	Relu relu3(100);
	Relu relu5(100);


		
	
	l6.weights->print();
	l6.bias->print();

	Sequential model;

	model.add_layer(&l0);
	model.add_layer(&relu1);
	model.add_layer(&l2);
	model.add_layer(&relu3);
	model.add_layer(&l4);
	model.add_layer(&relu5);
	model.add_layer(&l6);


	Mse mse;
	SGD opt = SGD();
	opt.lr = 0.01f;
	opt.model = &model;

	int epoch = 7;

	auto pgs = std::vector<unsigned int>{ 1,1 };
	tensor pg(pgs);
	pg.grad[0] = 1.0f;



	for (int i = 0; i < epoch; i++) {
	
		float tloss = 0.0f;
		for (int j = 0; j < batch; j++) {

			auto in = x[j];
			auto t = y[j];

			auto pred = model.forward(in, batch_size);
			auto loss = mse.forward(pred, t, batch_size);

			tloss += loss->value[0];
			loss->print();
			loss->backward(&pg);
			opt.step();
			model.zero_grad();
			pg.zero_grad();
			in->zero_grad();
			delete loss;
		}

		std::cout <<"echo: "<<i<< " tloss: " <<tloss<< std::endl;

	
	}

	printf("\n============= test ================\n");
	float begin  = 0.0f;

	auto test_s  = std::vector<unsigned int>{ 100,1 };
	tensor test(test_s) ;

	for (int i = 0; i < 100; i++) {
		test.value[i] = begin;
		begin = begin + 0.01;
	}

	auto tp = model.forward(&test, 100);

	for (int i = 0; i < 100; i++) {
		printf("%f,", tp->value[i]);
	}



	tp->print();

}



__global__
void hello_cuda(float* value) {

	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;


	printf(" grid_x : %d , grid_y : %d, thread_x : %d , thread_y : %d, row : %d , col : %d \n ",
		blockIdx.x, blockIdx.y,
		threadIdx.x, threadIdx.y,
		row, col
	);


	atomicAdd(value, 1.0f);

}


int main()
{
	train();
}


/*


	dim3 n_threads(2, 2);
	dim3 n_blocks(2, 2);

	float* p_count;
	cudaMallocManaged(&p_count,  sizeof(float)); //申请统一管理内存，可以直接写
	*p_count = 0.0f;

	hello_cuda << <n_blocks, n_threads >> > (p_count);
	cudaDeviceSynchronize(); //等待kernel执行完成

	printf("call count %f\n", *p_count);


	print_cuda_info();

*/