#include<stdio.h>
#include<string>
#include<random>
#include"network.h"
network_layer::network_layer(std::string act_func, int in_num, int out_num) :
	act_func(act_func), in_num(in_num), out_num(out_num) {
	W = new double *[this->out_num];
	for (int i = 0; i < this->out_num; i++)W[i] = new double[this->in_num];
	out_put = new double[this->out_num];
	b = new double[this->out_num];
	delta = new double[this->out_num];
	act_func_char = act_func[0];
	double limit = sqrt(6 / (double)(in_num + out_num));
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> randlimit(-limit, limit);
	for (int i = 0; i < out_num; i++) {
		b[i] = randlimit(mt);
		for (int j = 0; j < in_num; j++) {
			W[i][j] = randlimit(mt);
		}
	}
}
network_layer::~network_layer() {
	for (int i = 0; i < out_num; i++)delete[] W[i];
	delete[] W;
	delete[] out_put;
	delete[] b;
	delete[] delta;
};

/*•`‰æ—pŠÖ”*/
void network_layer::show_Wb() {
	for (int i = 0; i < out_num; i++) {
		for (int j = 0; j < in_num; j++) printf("%f ", W[i][j]);
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < out_num; i++) printf("%f ", b[i]);
	printf("\n\n");
}