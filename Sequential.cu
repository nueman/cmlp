#include "Sequential.h"

void Sequential::add_layer(Module* l) {
	layers.push_back(l);
}


void Sequential::update(float lr) {

	for (int i = 0; i < layers.size(); i++) {
		Module* layer = layers[i];
		layer->update(lr);
	}


}



void Sequential::zero_grad() {
	for (unsigned int i=0; i < this->l_outs.size(); i++) {
		delete  this->l_outs[i];
	}
	this->l_outs.clear();

	for (int i = 0; i < layers.size(); i++) {
		Module* layer = layers[i];
		layer->zero_grad();
	}


};



tensor*  Sequential::forward(tensor* in, unsigned int bs){

	check_shape(in, bs);

	tensor* tmp_in = in;

	for (int i = 0; i < layers.size(); i++) {
		Module* layer = layers[i];
		tensor*  out = layer->forward(tmp_in , bs);
	
		out->name ="e" +  std::to_string(i+1) ;
		tmp_in = out;

		if (i < layers.size() - 1) {
			this->l_outs.push_back(out);
		}
		
	}

	return tmp_in;

}
