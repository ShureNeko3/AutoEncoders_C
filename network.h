class network_layer {
public:
	std::string act_func;
	char act_func_char;
	int in_num;//���͐�
	int out_num;//�o�͐�
	double **W;//�d��
	double *b;//�o�C�A�X
	double *out_put;//�o�͒l
	double *delta;//�p�����[�^�X�V�p�ϐ�
	network_layer(std::string act_func, int in_num, int out_num);
	~network_layer();
	void show_Wb();
};