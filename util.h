#pragma once
#include<memory>



#include<vector>
#include<string>
#include <cmath>
#include <iostream>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<fstream>
#include <functional> 

#include <ctime>
#include <cmath>




#define block_size 32


void print_cuda_info();
std::vector<std::string> split(std::string s, std::string delimiter);