#pragma once
#include "module.h"

class optimizer{
public:
	Module* model;
	float lr = 0.1f;

	virtual void step() {};
};