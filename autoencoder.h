class AutoEncoder {
private:
	Data_Package *Data;//�f�[�^
	int enc_layers_num;//�G���R�[�_�w�̐�
	int dec_layers_num;//�f�R�[�_�w�̐�
	double train_ratio;//�w�K��
	std::string loss_type;//�����֐��̎��
	char loss_type_char;
	int batch_size;
	int batch_sample_num;
public:
	network_layer **enc_layers;
	network_layer **dec_layers;
	double *y;//�l�b�g���[�N�o�͒l
	double *t;//���t�f�[�^
	double *E_y;//�덷
	double ***sum_enc_delta_W;
	double **sum_enc_delta_b;
	double ***sum_dec_delta_W;
	double **sum_dec_delta_b;
	double ***enc_W_h;
	double ***dec_W_h;
	double **enc_b_h;
	double **dec_b_h;
	int **batch_index;
	int *shuffle;

	AutoEncoder(Data_Package *Data, int enc_layers_num, network_layer **enc_layers, int dec_layers_num, network_layer **dec_layers, double train_ratio, std::string loss_type, int batch_size, int batch_sample_num);
	~AutoEncoder();

	void set_train_batch();
	/*�v�Z�p�֐�*/
	void act_func_dotWb(network_layer *layer, double ** input_data, int data_index);
	void act_func_dotWb(network_layer *layer, network_layer *prev);

	/*�G���R�[�h�C�f�R�[�h�C�G���R�[�h&�f�R�[�h*/
	void encode(int data_index);
	void decode();
	void encode_decode(int data_index);

	/*�����֐�*/
	double calc_loss(int data_index);

	/*�p�����[�^�X�V*/
	void updata_parameters(double train_ratio, int batch_size);

	/*���O�w�K*/
	void pretuning();

	/*���t����w�K*/
	void finetuning();

	/*�`��p�֐�*/
	void print_Wb();
	void print_output(int data_index);
	void print_loss(int batch_size);
	void print_index_loss(int data_index);
	void plot_graph();
	void save_Wb();
};