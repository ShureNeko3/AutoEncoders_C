class Data_Package {
private:
	int data_type;//�f�[�^�̎��
	int output_num;//�o�̓f�[�^�̎��
	int sequence_elements;//���n��f�[�^�̎��ԕ�

public:
	int data_size;//�f�[�^�̃T�C�Y
	int input_num;//���͑w�̃m�[�h��
	Data_Package(char data_name[],int data_size, int data_type, int input_num, int output_num, int sequence_elements,double pos_amp,double force_amp);
	~Data_Package(void);
	void Show_Data();
	void Show_Ans();
	double **load_data;//���f�[�^
	double **input_data;//���͗p�ɉ��H�����f�[�^
	double **ans_data;//�o�͗p(���t�p)�ɉ��H�����f�[�^
};