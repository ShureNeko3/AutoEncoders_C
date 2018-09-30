#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string>
#include<iostream>
#include<random>
#include<time.h>
#include"utils.h"
#include"network.h"
#include"autoencoder.h"

int main() {
	//**************************//
	//**  ���[�U�ݒ�ϐ��ꗗ  **//
	//**************************//

	/*�f�[�^�ɂ��Ă̐ݒ�ϐ�*/
	const int sequence_elements = 30;//���n��f�[�^�̎��ԕ�
	const int elements_num = 6;//���̓f�[�^�̎��(�ʒu*3+��*3=6)
	const int input_num = sequence_elements*elements_num;//���͑w�̃m�[�h��
	const int output_num = 2;//�o�̓f�[�^�̎��(���x+��=2)
	const int data_size = 1000;//�f�[�^�̃T�C�Y
	const int data_type = 7;//�f�[�^�̎��(����+�ʒu*3+��*3=7)

	/*�w�ɂ��Ă̐ݒ�ϐ�*/
	const int latent_degree = 2;//���݋�Ԃ̎���
	const int enc_layers_num = 3;//�G���R�[�_�̑w�̐�(�C��)
	std::string enc_act_func[enc_layers_num] = { "relu","relu","linear" };//�������֐�("linear"or"sigmoid"or"tanh"or"relu")
	int enc_nodes[enc_layers_num + 1] = { input_num,90,30,latent_degree };//(���͑w+)���ԑw�̃m�[�h��
	const int dec_layers_num = 3;//�f�R�[�_�̑w�̐�(�C��)
	std::string dec_act_func[dec_layers_num] = { "relu","relu","linear" };//�������֐�("linear"or"sigmoid"or"tanh"or"relu")
	int dec_nodes[dec_layers_num + 1] = { latent_degree,30,90,input_num };

	/*�w�K�ɂ��Ă̐ݒ�ϐ�*/
	double train_ratio = 0.01;//�w�K��(���݂�"AdaGrad")
	std::string loss_type = "mse";//�����֐�("mse"(=mean square error))
	int epoch = 10;
	int batch_size = 100;
	int batch_sample_num = data_size / batch_size;

	/*�I�u�W�F�N�g�̏�����*/
	Data_Package *Data;
	char data_name[] = "./data/DATA_PROCESSED1_closed.dat";
	double pos_amp = 1;
	double force_amp = 1;
	Data = new Data_Package(data_name, data_size, data_type, input_num, output_num, sequence_elements, pos_amp, force_amp);
	//Data->Show_Data();
	network_layer **enc_layers = new network_layer*[enc_layers_num];
	network_layer **dec_layers = new network_layer*[dec_layers_num];
	for (int i = 0; i < enc_layers_num; i++) enc_layers[i] = new network_layer(enc_act_func[i], enc_nodes[i], enc_nodes[i + 1]);
	for (int i = 0; i < dec_layers_num; i++) dec_layers[i] = new network_layer(dec_act_func[i], dec_nodes[i], dec_nodes[i + 1]);

	AutoEncoder *AE;
	AE = new AutoEncoder(Data, enc_layers_num, enc_layers, dec_layers_num, dec_layers, train_ratio, loss_type, batch_size, batch_sample_num);

	clock_t start = clock();
	for (int i = 0; i < epoch; i++) {
		AE->set_train_batch();
		AE->updata_parameters(train_ratio, batch_size);
		printf("epoch = %d\n", i + 1);
		AE->print_loss(batch_size);
	}
	clock_t end = clock();
	std::cout << "learning time = " << (double)(end - start) / CLOCKS_PER_SEC << "sec" << std::endl;

	AE->plot_graph();
	AE->save_Wb();

	/*�I�u�W�F�N�g�̌㏈��*/
	delete AE;
	for (int i = 0; i < enc_layers_num; i++) delete enc_layers[i];
	for (int i = 0; i < dec_layers_num; i++) delete dec_layers[i];
	delete[] enc_layers;
	delete[] dec_layers;
	delete Data;
}