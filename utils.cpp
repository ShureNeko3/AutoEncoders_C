#include<stdio.h>
#include"utils.h"
Data_Package::Data_Package(char data_name[],int data_size, int data_type, int input_num, int output_num, int sequence_elements,double pos_amp,double force_amp) :
	data_size(data_size), data_type(data_type), input_num(input_num), output_num(output_num), sequence_elements(sequence_elements), load_data(NULL), input_data(NULL), ans_data(NULL) {
	load_data = new double*[data_size];
	for (int i = 0; i < data_size; i++) load_data[i] = new double[data_type];
	input_data = new double*[data_size];
	for (int i = 0; i < data_size; i++)input_data[i] = new double[input_num];
	ans_data = new double*[data_size];
	for (int i = 0; i < data_size; i++)ans_data[i] = new double[output_num];

	FILE *in_file;
	in_file = fopen(data_name, "r");
	int data_num = 0;
	while (fscanf(in_file, "%lf", &load_data[data_num][0]) != EOF && fscanf(in_file, "%lf", &load_data[data_num][1]) != EOF&&
		fscanf(in_file, "%lf", &load_data[data_num][2]) != EOF&&fscanf(in_file, "%lf", &load_data[data_num][3]) != EOF&&
		fscanf(in_file, "%lf", &load_data[data_num][4]) != EOF&&fscanf(in_file, "%lf", &load_data[data_num][5]) != EOF&&
		fscanf(in_file, "%lf", &load_data[data_num][6]) != EOF) {
		data_num++;
		if (data_num >= data_size)break;
	}
	fclose(in_file);
	for (int i = 0; i < data_size - sequence_elements; i++) {
		for (int j = 0; j < sequence_elements; j++) {
			for (int k = 0; k < 6; k++) {
				if (k / 3 == 0) input_data[i][j * 6 + k] = pos_amp*load_data[i + j][k + 1];
				else input_data[i][j * 6 + k] = force_amp*load_data[i + j][k + 1];

			}
		}
	}

	for (int i = data_size - sequence_elements; i < data_size; i++) {
		for (int j = 0; j < sequence_elements; j++) {
			for (int k = 0; k < 6; k++) {
				if (k / 3 == 0) input_data[i][j * 6 + k] = input_data[data_size - sequence_elements - 1][j * 6 + k];
				else input_data[i][j * 6 + k] = input_data[data_size - sequence_elements - 1][j * 6 + k];
			}
		}
	}
}
Data_Package::~Data_Package() {
	for (int i = 0; i < data_size; i++) delete[] load_data[i];
	for (int i = 0; i < data_size; i++)delete[] input_data[i];
	for (int i = 0; i < data_size; i++)delete[] ans_data[i];
	delete[] load_data;
	delete[] input_data;
	delete[] ans_data;
}
/*•`‰æ—pŠÖ”*/
void Data_Package::Show_Data() {
	for (int i = 0; i < data_size; i++) {
		for (int j = 0; j < data_type; j++) {
			printf("%f ", load_data[i][j]);
		}
		printf("\n");
	}
	for (int i = 0; i < data_size; i++) {
		for (int j = 0; j < input_num; j++) {
			printf("%f ", input_data[i][j]);
		}
		printf("\n");
	}
}
void Data_Package::Show_Ans() {
	for (int i = 0; i < data_size; i++) {
		printf("%f %f", ans_data[i][0], ans_data[i][1]);
		printf("\n");
	}
}