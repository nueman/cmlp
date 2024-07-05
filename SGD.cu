#include "SGD.h"

void SGD::step() {

	model->update(lr);

}