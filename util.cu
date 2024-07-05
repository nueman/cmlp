#include "util.h"


void print_cuda_info() {

	int dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);

	std::cout << "SM 数：" << devProp.multiProcessorCount << std::endl;
	std::cout << "warp大小：" << devProp.warpSize << std::endl;
	std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / devProp.warpSize << std::endl;
}



std::vector<std::string> split(std::string s, std::string delimiter) {
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}


	res.push_back(s.substr(pos_start));
	return res;
}

