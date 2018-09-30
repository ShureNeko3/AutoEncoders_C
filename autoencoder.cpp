#include<stdio.h>
#include<string>
#include<random>
#include"utils.h"
#include"network.h"
#include"autoencoder.h"

AutoEncoder::AutoEncoder(Data_Package *Data, int enc_layers_num, network_layer **enc_layers, int dec_layers_num, network_layer **dec_layers, double train_ratio, std::string loss_type, int batch_size, int batch_sample_num)
	:Data(Data), enc_layers_num(enc_layers_num), enc_layers(enc_layers), dec_layers_num(dec_layers_num), dec_layers(dec_layers), train_ratio(train_ratio), loss_type(loss_type), batch_size(batch_size), batch_sample_num(batch_sample_num) {
	y = new double[Data->input_num];
	t = new double[Data->input_num];
	E_y = new double[Data->input_num];
	sum_enc_delta_W = new double**[enc_layers_num];
	enc_W_h = new double**[enc_layers_num];
	for (int i = 0; i < enc_layers_num; i++) {
		sum_enc_delta_W[i] = new double*[enc_layers[i]->out_num];
		enc_W_h[i] = new double*[enc_layers[i]->out_num];
		for (int j = 0; j < enc_layers[i]->out_num; j++) {
			sum_enc_delta_W[i][j] = new double[enc_layers[i]->in_num];
			enc_W_h[i][j] = new double[enc_layers[i]->in_num];
		}
	}
	sum_enc_delta_b = new double*[enc_layers_num];
	enc_b_h = new double*[enc_layers_num];
	for (int i = 0; i < this->enc_layers_num; i++) {
		sum_enc_delta_b[i] = new double[enc_layers[i]->out_num];
		enc_b_h[i] = new double[enc_layers[i]->out_num];
	}
	sum_dec_delta_W = new double**[dec_layers_num];
	dec_W_h = new double**[dec_layers_num];
	for (int i = 0; i < this->dec_layers_num; i++) {
		sum_dec_delta_W[i] = new double*[dec_layers[i]->out_num];
		dec_W_h[i] = new double*[dec_layers[i]->out_num];
		for (int j = 0; j < dec_layers[i]->out_num; j++) {
			sum_dec_delta_W[i][j] = new double[dec_layers[i]->in_num];
			dec_W_h[i][j] = new double[dec_layers[i]->in_num];
		}
	}
	sum_dec_delta_b = new double*[dec_layers_num];
	dec_b_h = new double*[dec_layers_num];
	for (int i = 0; i < this->dec_layers_num; i++) {
		sum_dec_delta_b[i] = new double[dec_layers[i]->out_num];
		dec_b_h[i] = new double[dec_layers[i]->out_num];
	}
	batch_index = new int*[this->batch_sample_num];
	for (int i = 0; i < this->batch_sample_num; i++)batch_index[i] = new int[this->batch_size];
	shuffle = new int[Data->data_size];
	loss_type_char = loss_type[0];
	for (int i = 0; i < enc_layers_num; i++) {
		for (int j = 0; j < enc_layers[i]->out_num; j++) {
			enc_b_h[i][j] = 0.000000010;
			for (int k = 0; k < enc_layers[i]->in_num; k++) {
				enc_W_h[i][j][k] = 0.000000010;
			}
		}
	}
	for (int i = 0; i < dec_layers_num; i++) {
		for (int j = 0; j < dec_layers[i]->out_num; j++) {
			dec_b_h[i][j] = 0.000000010;
			for (int k = 0; k < dec_layers[i]->in_num; k++) {
				dec_W_h[i][j][k] = 0.000000010;
			}
		}
	}
}
AutoEncoder::~AutoEncoder() {
	//delete[] y;
	//delete[] t;
	delete[] E_y;
	for (int i = 0; i < enc_layers_num; i++) {
		for (int j = 0; j < enc_layers[i]->out_num; j++) {
			delete[] sum_enc_delta_W[i][j];
			delete[] enc_W_h[i][j];
		}
		delete[] sum_enc_delta_W[i];
		delete[] enc_W_h[i];
	}
	delete[] sum_enc_delta_W;
	delete[] enc_W_h;
	for (int i = 0; i < enc_layers_num; i++) {
		delete[] sum_enc_delta_b[i];
		delete[] enc_b_h[i];
	}
	delete[] sum_enc_delta_b;
	delete[] enc_b_h;
	for (int i = 0; i < dec_layers_num; i++) {
		for (int j = 0; j < dec_layers[i]->out_num; j++) {
			delete[] sum_dec_delta_W[i][j];
			delete[] dec_W_h[i][j];
		}
		delete[] sum_dec_delta_W[i];
		delete[] dec_W_h[i];
	}
	delete[] sum_dec_delta_W;
	delete[] dec_W_h;
	for (int i = 0; i < dec_layers_num; i++) {
		delete[] sum_dec_delta_b[i];
		delete[] dec_b_h[i];
	}
	delete[] sum_dec_delta_b;
	delete[] dec_b_h;
	for (int i = 0; i < batch_sample_num; i++) {
		delete[] batch_index[i];
	}
	delete[] batch_index;
	delete[] shuffle;
}

void AutoEncoder::set_train_batch() {
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_int_distribution<> rand(0, Data->data_size - 1);
	for (int i = 0; i < Data->data_size; i++) {
		shuffle[i] = i;
	}
	for (int i = 0; i < Data->data_size; i++) {
		int rnd = rand(mt);
		int temp = shuffle[i];
		shuffle[i] = shuffle[rnd];
		shuffle[rnd] = temp;
	}
	for (int i = 0; i < batch_sample_num; i++) {
		for (int j = 0; j < batch_size; j++) {
			batch_index[i][j] = shuffle[i*batch_size + j];
		}
	}
}

/*計算用関数*/
void AutoEncoder::act_func_dotWb(network_layer *layer, double ** input_data, int data_index) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * input_data[data_index][j];
		}
		if (layer->act_func_char == 'l');
		else if (layer->act_func_char == 's')layer->out_put[i] = 1 / (1 + exp(-layer->out_put[i]));
		else if (layer->act_func_char == 't')layer->out_put[i] = tanh(layer->out_put[i]);
		else if (layer->act_func_char == 'r'&&layer->out_put[i] <= 0)layer->out_put[i] = 0;
	}
}
void AutoEncoder::act_func_dotWb(network_layer *layer, network_layer *prev) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < prev->out_num; j++) {
			layer->out_put[i] += layer->W[i][j] * prev->out_put[j];
		}
		if (layer->act_func_char == 'l');
		else if (layer->act_func_char == 's')layer->out_put[i] = 1 / (1 + exp(-layer->out_put[i]));
		else if (layer->act_func_char == 't')layer->out_put[i] = tanh(layer->out_put[i]);
		else if (layer->act_func_char == 'r'&&layer->out_put[i] <= 0)layer->out_put[i] = 0;
	}
}

/*エンコード，デコード，エンコード&デコード*/
void AutoEncoder::encode(int data_index) {
	for (int i = 0; i < enc_layers_num; i++) {
		if (i == 0) act_func_dotWb(enc_layers[i], Data->input_data, data_index);
		else act_func_dotWb(enc_layers[i], enc_layers[i - 1]);
	}
}
void AutoEncoder::decode() {
	for (int i = 0; i < dec_layers_num; i++) {
		if (i == 0)act_func_dotWb(dec_layers[i], enc_layers[enc_layers_num - 1]);
		else act_func_dotWb(dec_layers[i], dec_layers[i - 1]);
	}
}
void AutoEncoder::encode_decode(int data_index) {
	for (int i = 0; i < enc_layers_num; i++) {
		if (i == 0) act_func_dotWb(enc_layers[i], Data->input_data, data_index);
		else act_func_dotWb(enc_layers[i], enc_layers[i - 1]);
	}
	for (int i = 0; i < dec_layers_num; i++) {
		if (i == 0)act_func_dotWb(dec_layers[i], enc_layers[enc_layers_num - 1]);
		else act_func_dotWb(dec_layers[i], dec_layers[i - 1]);
	}
}

/*損失関数*/
double AutoEncoder::calc_loss(int data_index) {
	double loss = 0;
	for (int i = 0; i < Data->input_num; i++) {
		if (loss_type == "mse") loss += pow(dec_layers[dec_layers_num - 1]->out_put[i] - Data->input_data[data_index][i], 2) / 2;
	}
	return loss;
}

/*パラメータ更新*/
void AutoEncoder::updata_parameters(double train_ratio, int batch_size) {
	int data_index;
	for (int sample = 0; sample < batch_sample_num; sample++) {

		for (int i = dec_layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < dec_layers[i]->in_num; j++) {
				for (int k = 0; k < dec_layers[i]->out_num; k++) {
					sum_dec_delta_W[i][k][j] = 0;
					if (j == 0)sum_dec_delta_b[i][k] = 0;
				}
			}
		}
		for (int i = enc_layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < enc_layers[i]->in_num; j++) {
				for (int k = 0; k < enc_layers[i]->out_num; k++) {
					sum_enc_delta_W[i][k][j] = 0;
					if (j == 0)sum_enc_delta_b[i][k] = 0;
				}
			}
		}

		for (int index = 0; index < batch_size; index++) {
			data_index = batch_index[sample][index];
			encode_decode(data_index);
			y = dec_layers[dec_layers_num - 1]->out_put;
			t = Data->input_data[data_index];
			for (int i = 0; i < Data->input_num; i++) {
				if (loss_type_char == 'm') E_y[i] = y[i] - t[i];
			}

			for (int i = dec_layers_num - 1; i >= 0; i--) {
				for (int j = 0, jj = dec_layers[i]->in_num; j < jj; j++) {
					for (int k = 0, kk = dec_layers[i]->out_num; k < kk; k++) {
						dec_layers[i]->delta[k] = 0;
						if (i != dec_layers_num - 1) {
							if (dec_layers[i]->act_func_char == 'l')for (int l = 0, ll = dec_layers[i + 1]->out_num; l < ll; l++) dec_layers[i]->delta[k] += dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
							else if (dec_layers[i]->act_func_char == 's')for (int l = 0, ll = dec_layers[i + 1]->out_num; l < ll; l++) dec_layers[i]->delta[k] += (1 - dec_layers[i]->out_put[k])*dec_layers[i]->out_put[k] * dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
							else if (dec_layers[i]->act_func_char == 't')for (int l = 0, ll = dec_layers[i + 1]->out_num; l < ll; l++) {
								dec_layers[i]->delta[k] += (1 - pow(dec_layers[i]->out_put[k], 2))* dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
							}
							else if (dec_layers[i]->act_func_char == 'r') {
								for (int l = 0, ll = dec_layers[i + 1]->out_num; l < ll; l++) {
									if (dec_layers[i]->out_put[k] > 0)dec_layers[i]->delta[k] += dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
								}
							}
						}
						else {
							if (dec_layers[i]->act_func_char == 'l') dec_layers[i]->delta[k] = E_y[k];
							else if (dec_layers[i]->act_func_char == 's')dec_layers[i]->delta[k] = (1 - y[k])*y[k] * E_y[k];
							else if (dec_layers[i]->act_func_char == 't') dec_layers[i]->delta[k] = (1 - pow(y[k], 2))* E_y[k];
							else if (dec_layers[i]->act_func_char == 'r') {
								if (y[k] > 0)dec_layers[i]->delta[k] = E_y[k];
								else dec_layers[i]->delta[k] = 0;
							}
						}
						if (i == 0) sum_dec_delta_W[i][k][j] += dec_layers[i]->delta[k] * enc_layers[enc_layers_num - 1]->out_put[j];
						else sum_dec_delta_W[i][k][j] += dec_layers[i]->delta[k] * dec_layers[i - 1]->out_put[j];
						if (j == 0) {
							if (i == 0) sum_dec_delta_b[i][k] += dec_layers[i]->delta[k];
							else sum_dec_delta_b[i][k] += dec_layers[i]->delta[k];
						}
					}
				}
			}

			for (int i = enc_layers_num - 1; i >= 0; i--) {
				for (int j = 0; j < enc_layers[i]->in_num; j++) {
					for (int k = 0; k < enc_layers[i]->out_num; k++) {
						enc_layers[i]->delta[k] = 0;
						if (i != enc_layers_num - 1) {
							if (enc_layers[i]->act_func_char == 'l')for (int l = 0, ll = enc_layers[i + 1]->out_num; l < ll; l++) enc_layers[i]->delta[k] += enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
							else if (enc_layers[i]->act_func_char == 's')for (int l = 0, ll = enc_layers[i + 1]->out_num; l < ll; l++) enc_layers[i]->delta[k] += (1 - enc_layers[i]->out_put[k])*enc_layers[i]->out_put[k] * enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
							else if (enc_layers[i]->act_func_char == 't')for (int l = 0, ll = enc_layers[i + 1]->out_num; l < ll; l++)enc_layers[i]->delta[k] += (1 - pow(enc_layers[i]->out_put[k], 2))* enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
							else if (enc_layers[i]->act_func_char == 'r') {
								for (int l = 0, ll = enc_layers[i + 1]->out_num; l < ll; l++) {
									if (enc_layers[i]->out_put[k] > 0)enc_layers[i]->delta[k] += enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
								}
							}
						}
						else {
							if (enc_layers[i]->act_func_char == 'l') for (int l = 0,ll=dec_layers[0]->out_num; l < ll; l++) enc_layers[i]->delta[k] += dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
							else if (enc_layers[i]->act_func_char == 's') for (int l = 0,ll=dec_layers[0]->out_num; l < ll; l++) enc_layers[i]->delta[k] += (1 - dec_layers[0]->out_put[k])*dec_layers[0]->out_put[k] * dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
							else if (enc_layers[i]->act_func_char == 't') for (int l = 0,ll=dec_layers[0]->out_num; l < ll; l++) enc_layers[i]->delta[k] += (1 - pow(dec_layers[0]->out_put[k], 2))* dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
							else if (enc_layers[i]->act_func_char == 'r') {
								for (int l = 0,ll=dec_layers[0]->out_num; l < ll; l++) {
									if (dec_layers[i]->out_put[k] > 0)enc_layers[i]->delta[k] += dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
								}
							}
						}
						if (i == 0) sum_enc_delta_W[i][k][j] += enc_layers[i]->delta[k] * Data->input_data[data_index][j];
						else sum_enc_delta_W[i][k][j] += enc_layers[i]->delta[k] * enc_layers[i - 1]->out_put[j];
						if (j == 0) {
							if (i == 0) sum_enc_delta_b[i][k] += enc_layers[i]->delta[k];
							else sum_enc_delta_b[i][k] += enc_layers[i]->delta[k];
						}
					}
				}
			}
		}

		for (int i = dec_layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < dec_layers[i]->in_num; j++) {
				for (int k = 0; k < dec_layers[i]->out_num; k++) {
					dec_W_h[i][k][j] += pow(sum_dec_delta_W[i][k][j], 2);
					dec_layers[i]->W[k][j] -= train_ratio*sum_dec_delta_W[i][k][j] / sqrt(dec_W_h[i][k][j]);
					//printf("%d %f %f\n", sample, dec_W_h[i][k][j], sqrt(dec_W_h[i][k][j]));
					if (j == 0) {
						dec_b_h[i][k] += pow(sum_dec_delta_b[i][k], 2);
						dec_layers[i]->b[k] -= train_ratio*sum_dec_delta_b[i][k] / sqrt(dec_b_h[i][k]);
					}
				}
			}
		}
		for (int i = enc_layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < enc_layers[i]->in_num; j++) {
				for (int k = 0; k < enc_layers[i]->out_num; k++) {
					enc_W_h[i][k][j] += pow(sum_enc_delta_W[i][k][j], 2);
					enc_layers[i]->W[k][j] -= train_ratio*sum_enc_delta_W[i][k][j] / sqrt(enc_W_h[i][k][j]);
					if (j == 0) {
						enc_b_h[i][k] += pow(sum_enc_delta_b[i][k], 2);
						enc_layers[i]->b[k] -= train_ratio*sum_enc_delta_b[i][k] / sqrt(enc_b_h[i][k]);
					}
				}
			}
		}
	}
}

/*事前学習*/
void AutoEncoder::pretuning() {
	for (int i = 0; i < Data->data_size; i++) {
		encode_decode(i);
		//updata_parameters(calc_loss(i), train_ratio, i);
	}
}

/*教師あり学習*/
void AutoEncoder::finetuning() {

}

/*描画用関数*/
void AutoEncoder::print_Wb() {
	for (int i = 0; i < enc_layers_num; i++) {
		printf("encoder layer %d Weight", i);
		for (int j = 0; j < enc_layers[i]->out_num; j++) {
			for (int k = 0; k < enc_layers[i]->in_num; k++) printf("%f ", enc_layers[i]->W[j][k]);
			printf("\n");
		}
		printf("\n");
		printf("encoder layer %d bias", i);
		for (int j = 0; j < enc_layers[i]->out_num; j++) printf("%f ", enc_layers[i]->b[j]);
		printf("\n\n");
	}
	for (int i = 0; i < dec_layers_num; i++) {
		printf("decoder layer %d Weight", i);
		for (int j = 0; j < dec_layers[i]->out_num; j++) {
			for (int k = 0; k < dec_layers[i]->in_num; k++) printf("%f ", dec_layers[i]->W[j][k]);
			printf("\n");
		}
		printf("\n");
		printf("decoder layer %d bias", i);
		for (int j = 0; j < dec_layers[i]->out_num; j++) printf("%f ", dec_layers[i]->b[j]);
		printf("\n\n");
	}
}
void AutoEncoder::print_output(int data_index) {
	printf("data_index=%d\n", data_index);
	for (int i = 0; i < Data->input_num; i++) {
		printf("%d %f  %f \n", i, dec_layers[dec_layers_num - 1]->out_put[i], Data->input_data[data_index][i]);
	}
	printf("\n");
}
void AutoEncoder::print_loss(int batch_size) {
	double loss = 0;
	int data_index;
	for (int sample = 0; sample < batch_sample_num; sample++) {
		for (int index = 0; index < batch_size; index++) {
			data_index = batch_index[sample][index];
			encode_decode(data_index);
			for (int i = 0; i < Data->input_num; i++) loss += pow(dec_layers[dec_layers_num - 1]->out_put[i] - Data->input_data[data_index][i], 2) / 2;
		}
	}
	printf("loss = %f\n", loss / (batch_sample_num*batch_size));
}
void AutoEncoder::print_index_loss(int data_index) {
	double loss = 0;
	for (int i = 0; i < Data->input_num; i++) if (loss_type == "mse") loss += pow(dec_layers[dec_layers_num - 1]->out_put[i] - Data->input_data[data_index][i], 2) / 2;
	printf("%d %f\n", data_index, loss);
}
void AutoEncoder::plot_graph() {
	FILE *gp;
	FILE *fp;
	fp = fopen("latent_space.dat", "w");
	for (int i = 0; i < Data->data_size; i++) {
		encode(i);
		fprintf(fp, "%f %f\r\n", enc_layers[enc_layers_num - 1]->out_put[0], enc_layers[enc_layers_num - 1]->out_put[1]);
	}
	fclose(fp);
	gp = _popen("gnuplot -persist", "w");
	fprintf(gp, "set terminal postscript enhanced color\nset output './figure/latent_space.eps'\n");
	fprintf(gp, "plot 'latent_space.dat' using 1:2\n");
	fprintf(gp, "quit\n");
	fflush(gp);
	_pclose(gp);
}
void AutoEncoder::save_Wb() {
	FILE *fp1, *fp2, *fp3, *fp4;
	fp1 = fopen("./weight/Encoder_Weight.dat", "w");
	fp2 = fopen("./weight/Encoder_Bias.dat", "w");
	fp3 = fopen("./weight/Decoder_Weight.dat", "w");
	fp4 = fopen("./weight/Decoder_Bias.dat", "w");
	fprintf(fp1, "%d\n", enc_layers_num);
	for (int i = 0; i < enc_layers_num; i++) {
		fprintf(fp1, "%d %d\n", enc_layers[i]->out_num, enc_layers[i]->in_num);
	}
	for (int i = 0; i < enc_layers_num; i++) {
		for (int j = 0; j < enc_layers[i]->out_num; j++) {
			for (int k = 0; k < enc_layers[i]->in_num; k++) {
				fprintf(fp1, " %f", enc_layers[i]->W[j][k]);
			}
			fprintf(fp1, "\r\n");
		}
	}
	fprintf(fp2, "%d\n", enc_layers_num);
	for (int i = 0; i < enc_layers_num; i++) {
		fprintf(fp2, "%d\n", enc_layers[i]->out_num);
	}
	for (int i = 0; i < enc_layers_num; i++) {
		for (int j = 0; j < enc_layers[i]->out_num; j++) {
			fprintf(fp2, " %f", enc_layers[i]->b[j]);
		}
		fprintf(fp2, "\r\n");
	}
	fprintf(fp3, "%d\n", dec_layers_num);
	for (int i = 0; i < dec_layers_num; i++) {
		fprintf(fp3, "%d %d\n", dec_layers[i]->out_num, dec_layers[i]->in_num);
	}
	for (int i = 0; i < dec_layers_num; i++) {
		for (int j = 0; j < dec_layers[i]->out_num; j++) {
			for (int k = 0; k < dec_layers[i]->in_num; k++) {
				fprintf(fp3, " %f", dec_layers[i]->W[j][k]);
			}
			fprintf(fp3, "\r\n");
		}
	}
	fprintf(fp4, "%d\n", dec_layers_num);
	for (int i = 0; i < dec_layers_num; i++) {
		fprintf(fp4, "%d\n", dec_layers[i]->out_num);
	}
	for (int i = 0; i < dec_layers_num; i++) {
		for (int j = 0; j < dec_layers[i]->out_num; j++) {
			fprintf(fp4, " %f", dec_layers[i]->b[j]);
		}
		fprintf(fp4, "\r\n");
	}

	fclose(fp1), fclose(fp2), fclose(fp3), fclose(fp4);
}