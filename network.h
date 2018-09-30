class network_layer {
public:
	std::string act_func;
	char act_func_char;
	int in_num;//入力数
	int out_num;//出力数
	double **W;//重み
	double *b;//バイアス
	double *out_put;//出力値
	double *delta;//パラメータ更新用変数
	network_layer(std::string act_func, int in_num, int out_num);
	~network_layer();
	void show_Wb();
};