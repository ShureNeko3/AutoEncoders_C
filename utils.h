class Data_Package {
private:
	int data_type;//データの種類
	int output_num;//出力データの種類
	int sequence_elements;//時系列データの時間幅

public:
	int data_size;//データのサイズ
	int input_num;//入力層のノード数
	Data_Package(char data_name[],int data_size, int data_type, int input_num, int output_num, int sequence_elements,double pos_amp,double force_amp);
	~Data_Package(void);
	void Show_Data();
	void Show_Ans();
	double **load_data;//生データ
	double **input_data;//入力用に加工したデータ
	double **ans_data;//出力用(教師用)に加工したデータ
};