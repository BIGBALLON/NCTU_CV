#include <opencv2\highgui\highgui.hpp>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <cmath>

/* namespace */
using namespace cv;
using namespace std;

/* Global variable */
const int PIC_MAX_NUM = 6;
int COL_MAX_NUM = 0;
int ROW_MAX_NUM = 0;
string pic_name = "bunny";
string light_file = "";

/* Matrix variable */
Mat imgs[6];
Mat lightsource, intensity;
Mat pseudo, b, normal, depth;

/* Matrix calculation */
void calc_intensity();                    // calculate intensity
void calc_normal();                       // calculate normal
void calc_depth();                        // calculate depth
inline void calc_b();                     // calculate b
void pseudo_inverse();                    // calculate (U_t * U )^-1 * U_t
void print_mat(Mat mat);                  // print mat -- dubug using

/* Pooling -- using while calc_depth */
void mean_pooling( Mat &mat );			  // kernal 3*3 cal mean
void max_pooling( Mat &mat );             // kernal 3*3 cal max
void pooling_v( Mat &mat );               // values correction

/* I/O operation */
bool redirection(int argc, char* argv[]); // redirection input & output files
bool load_img();                          // load image file
bool load_light_source();                 // load lightsoucre form file
void output();                            // output to *.ply

int main(int argc, char* argv[]) {

	redirection(argc, argv);
	load_img();
	load_light_source();
	calc_intensity();
	pseudo_inverse();
	calc_b();
	calc_normal();
	calc_depth();
	output();

	return 0;
}

bool redirection(int argc, char* argv[]) {
	if (argc > 1) {
		pic_name = string(argv[1]);
	}
	string file_name = "./test/" + pic_name + "/LightSource.txt";
	string output_name = pic_name + ".ply";
	freopen(file_name.c_str(), "r", stdin);
	freopen(output_name.c_str(), "w", stdout);
	return true;
}

void print_mat(Mat mat) {
	for (int i = 0; i < mat.rows; i++) {
		const double* mati = mat.ptr<double>(i);
		for (int j = 0; j < mat.cols; j++) {
			printf("%f ", mati[j]);
		}
		printf("\n");
	}
	printf("\n");
}

bool load_img() {
	for (int i = 1; i <= PIC_MAX_NUM; ++i) {
		stringstream ss;
		ss << i;
		string pic_num = ss.str();
		string file_name = "./test/" + pic_name + "/pic" + pic_num + ".bmp";
		imgs[i - 1] = imread(file_name, IMREAD_GRAYSCALE);
	}
	COL_MAX_NUM = imgs[0].cols;
	ROW_MAX_NUM = imgs[0].rows;
	return true;
}

bool load_light_source() {
	int i;
	double x, y, z;
	lightsource = Mat(Mat::zeros(6, 3, CV_64FC(1)));
	while (~scanf("pic%d: (%lf,%lf,%lf)\n", &i, &x, &y, &z)) {
		Mat tmp = (Mat_<double>(1, 3) << x, y, z);
		tmp.copyTo(lightsource.row(i - 1));
	}
	return true;
}

void calc_intensity() {
	int size = COL_MAX_NUM * ROW_MAX_NUM;
	intensity = Mat(Mat::zeros(6, size, CV_64FC(1)));
	for (int k = 0; k < PIC_MAX_NUM; k++) {
		for (int i = 0; i < ROW_MAX_NUM; i++) {
			for (int j = 0; j < COL_MAX_NUM; j++) {
				intensity.at<double>(k, i * COL_MAX_NUM + j) = imgs[k].at<uchar>(i, j);
			}
		}
	}
}

void pseudo_inverse() {
	Mat u = lightsource;
	Mat ut = lightsource.t();
	pseudo = (ut * u).inv(CV_SVD) * ut;
}

inline void calc_b() {
	b = pseudo * intensity;
}

void calc_normal() {
	int size = COL_MAX_NUM * ROW_MAX_NUM;
	normal = Mat(Mat::zeros(3, size, CV_64FC(1)));
	for (int i = 0; i < size; ++i) {
		double b_dis = sqrt(pow(b.col(i).at<double>(0, 0), 2)
			+ pow(b.col(i).at<double>(1, 0), 2)
			+ pow(b.col(i).at<double>(2, 0), 2));
		if (b_dis != 0) {
			Mat tmp = b.col(i) / b_dis;
			tmp.copyTo(normal.col(i));
		}
		else {
			Mat tmp(Mat::zeros(3, 1, CV_64FC(1)));
			tmp.copyTo(normal.col(i));
		}
	}
}

void mean_pooling( Mat &mat ){
	for (int i = 1; i < ROW_MAX_NUM - 1; i++) {
		for (int j = 1; j < COL_MAX_NUM - 1; j++) {
			double tmp = (mat.at<double>(i, j) + mat.at<double>(i - 1, j) + mat.at<double>(i, j - 1)
				+ mat.at<double>(i - 1, j - 1) + mat.at<double>(i + 1, j) + mat.at<double>(i, j + 1)
				+ mat.at<double>(i + 1, j + 1) + mat.at<double>(i - 1, j + 1) + mat.at<double>(i + 1, j - 1)) / 9.0;
			mat.at<double>(i, j) = tmp;
		}
	}
}

void max_pooling( Mat &mat ){
	for (int i = 1; i < ROW_MAX_NUM - 1; i++) {
		for (int j = 1; j < COL_MAX_NUM - 1; j++) {
			double tmp = 0.0;
			tmp = max(tmp,mat.at<double>(i, j));
			tmp = max(tmp,mat.at<double>(i+1, j));
			tmp = max(tmp,mat.at<double>(i, j+1));
			tmp = max(tmp,mat.at<double>(i+1, j+1));
			tmp = max(tmp,mat.at<double>(i-1, j));
			tmp = max(tmp,mat.at<double>(i, j-1));
			tmp = max(tmp,mat.at<double>(i-1, j-1));
			tmp = max(tmp,mat.at<double>(i-1, j+1));
			tmp = max(tmp,mat.at<double>(i+1, j-1));
			mat.at<double>(i, j) = tmp;
		}
	}	
}
void pooling_v( Mat &mat ){
	double tmp = 0.0, mean = 0.0;
	int count = 0;
	for (int i = 0; i < ROW_MAX_NUM; ++i){
		for (int j = 0; j < COL_MAX_NUM; ++j){
			if( mat.at<double>(i, j) < 0 ){
				tmp += mat.at<double>(i, j);
				count++;
			}
		}
	}
	mean = tmp / count;
	for (int i = 0; i < ROW_MAX_NUM; ++i){
		for (int j = 0; j < COL_MAX_NUM; ++j){
			if( mat.at<double>(i, j) > 0 ){
				mat.at<double>(i, j) = mean;
			}
		}
	}
}
void calc_depth() {
	depth = Mat(Mat::zeros(ROW_MAX_NUM, COL_MAX_NUM, CV_64FC(1)));
	for (int i = 1; i < ROW_MAX_NUM; i++) {
		Mat current = normal.col(i * COL_MAX_NUM);
		if (current.at<double>(2, 0) != 0) {
			depth.at<double>(i, 0) = (-1) * current.at<double>(1, 0)
				/ current.at<double>(2, 0) + depth.at<double>(i - 1, 0);
		}
		else {
			depth.at<double>(i, 0) = 0;
		}
	}
	for (int i = 0; i < ROW_MAX_NUM; i++) {
		for (int j = 1; j < COL_MAX_NUM; j++) {
			Mat current = normal.col(i * COL_MAX_NUM + j);
			if (current.at<double>(2, 0) != 0) {
				depth.at<double>(i, j) = (-1) * current.at<double>(0, 0) / current.at<double>(2, 0)
					+ depth.at<double>(i, j - 1);
			}
			else {
				depth.at<double>(i, 0) = 0;
			}
		}
	}
	mean_pooling(depth);
	Mat depth_2 = Mat::zeros(ROW_MAX_NUM, COL_MAX_NUM, CV_64FC(1));
	for (int i = 1; i < COL_MAX_NUM; i++) {
		Mat current = normal.col(i * ROW_MAX_NUM);
		if (current.at<double>(2, 0) != 0) {
			depth_2.at<double>(i, 0) = (-1) * current.at<double>(1, 0)
				/ current.at<double>(2, 0) + depth_2.at<double>(i - 1, 0);
		}
		else {
			depth_2.at<double>(i, 0) = 0;
		}
	}
	for (int i = 0; i < COL_MAX_NUM; i++) {
		for (int j = 1; j < ROW_MAX_NUM; j++) {
			Mat current = normal.col(i * ROW_MAX_NUM + j);
			if (current.at<double>(2, 0) != 0) {
				depth_2.at<double>(i, j) = (-1) * current.at<double>(0, 0) / current.at<double>(2, 0)
					+ depth_2.at<double>(i, j - 1);
			}
			else {
				depth_2.at<double>(i, 0) = 0;
			}
		}
	}
	mean_pooling(depth_2);

	for (int i = 0; i < ROW_MAX_NUM; i++) {
		for (int j = 0; j < COL_MAX_NUM; j++) {
			depth.at<double>(i, j) = (depth.at<double>(i, j) + depth_2.at<double>(i, j)) / 2;
		}
	}
}

void output() {

	printf("ply\nformat ascii 1.0\n");
	printf("comment alpha=%.1f\n", 1.0);
	printf("element vertex %d\n", ROW_MAX_NUM * COL_MAX_NUM);
	printf("property float x\nproperty float y\nproperty float z\n");
	printf("property uchar red\nproperty uchar green\nproperty uchar blue z\n");
	printf("end_header\n");
	for (int i = 0; i < ROW_MAX_NUM; i++) {
		for (int j = 0; j < COL_MAX_NUM; j++) {
			printf("%d %d %.5f 255 255 255\n", i, j, depth.at<double>(i, j));
		}
	}
}