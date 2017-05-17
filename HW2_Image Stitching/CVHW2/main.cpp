#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <stdio.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cmath>
#include <io.h>

using namespace cv;

#define MAX_PIC_SIZE            10
#define MAX_K                   2
#define RANSAC_ROUND			3000				    // round of RANSAC
#define MAX_POINT               2048

std::string image_name;
std::vector < KeyPoint>keypoints[MAX_PIC_SIZE]; 
std::vector < KeyPoint> keypoint_sample;
std::vector < KeyPoint> keypoint_target;
std::vector < int > index_table[MAX_POINT];
int pic_num;

int hash[MAX_POINT];

// warp matrix
Mat homogrphy[MAX_PIC_SIZE];
Mat homogrphy_sample_to_target;

// image matrix
Mat image_sample;
Mat image_target;
std::vector<Mat> image_puzzles;

// descriptor matrix
Mat descriptor_sample;
Mat descriptor_target;
Mat descriptor_puzzles[MAX_PIC_SIZE];

bool cmp(std::pair<int, float> x, std::pair<int, float> y) {
	return x.second < y.second;
}

bool load_image();
bool cal_sift(std::vector < KeyPoint>& keypoints_1, std::vector < KeyPoint>& keypoints_2, Mat& img_1, Mat& img_2, Mat& descriptor_1, Mat& descriptor_2);
bool knn(Mat base, Mat target);
void printde() { std::cout << "debug" << std::endl; }
void printMat(Mat M) {
	for (size_t i = 0; i < M.rows; i++) {
		for (size_t j = 0; j < M.cols; j++) {
			std::cout << M.at<float>(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void random_select( int id[5], int size, std::vector < KeyPoint> kp_base) {
	memset(hash, 0, sizeof(hash));
	for (size_t i = 1; i <= 4; i++) {
		int rand_num = rand() % size;
		while (true) {
			rand_num = rand() % size;
			if (kp_base[rand_num].pt.x == 0 && kp_base[rand_num].pt.y == 0) continue;
			if (hash[rand_num] != 0) continue;
			else break;
		}
		hash[rand_num] = 1;
		id[i] = rand_num;
	}
}

Mat construct_mat( int X1, int X2, int X3, int X4, int Y1, int Y2, int Y3, int Y4, int x1, int x2, int x3, int x4, int y1, int y2, int y3, int y4 ) {
	Mat A = Mat(8, 9, CV_32F);
	float data1[] = { X1, Y1, 1, 0, 0, 0, -x1*X1, -x1*Y1, -x1 };
	Mat(1, 9, CV_32FC1, data1).copyTo(A.row(0));
	float data2[] = { 0, 0, 0, X1, Y1, 1, -y1*X1, -y1*Y1, -y1 };
	Mat(1, 9, CV_32FC1, data2).copyTo(A.row(1));
	float data3[] = { X2, Y2, 1, 0, 0, 0, -x2*X2, -x2*Y2, -x2 };
	Mat(1, 9, CV_32FC1, data3).copyTo(A.row(2));
	float data4[] = { 0, 0, 0, X2, Y2, 1, -y2*X2, -y2*Y2, -y2 };
	Mat(1, 9, CV_32FC1, data4).copyTo(A.row(3));
	float data5[] = { X3, Y3, 1, 0, 0, 0, -x3*X3, -x3*Y3, -x3 };
	Mat(1, 9, CV_32FC1, data5).copyTo(A.row(4));
	float data6[] = { 0, 0, 0, X3, Y3, 1, -y3*X3, -y3*Y3, -y3 };
	Mat(1, 9, CV_32FC1, data6).copyTo(A.row(5));
	float data7[] = { X4, Y4, 1, 0, 0, 0, -x4*X4, -x4*Y4, -x4 };
	Mat(1, 9, CV_32FC1, data7).copyTo(A.row(6));
	float data8[] = { 0, 0, 0, X4, Y4, 1, -y4*X4, -y4*Y4, -y4 };
	Mat(1, 9, CV_32FC1, data8).copyTo(A.row(7));

	//printMat(A);
	Mat eigen_value = Mat::zeros(8, 9, CV_32FC1);
	Mat eigen_vector = Mat::zeros(8, 9, CV_32FC1);
	A = A.t()*A;
	eigen(A, eigen_value, eigen_vector);

	Mat H = eigen_vector.row(8);
	H = H.reshape(0, 3);
	//printMat(H);
	return H;
}

void ransac(std::vector < KeyPoint> kp_base, std::vector<KeyPoint> kp_target, Mat& _best_affine, int target = 0) {
	int base_id[5], target_id[5][5];
	int max_inlier = 0;

	for (size_t cas = 0; cas < 1500; cas++){
		if ((cas + 1) % 500 == 0) {
			printf("iteration: %d\n", cas+1);
		}
		random_select(base_id, kp_base.size(), kp_base);

		for (size_t i = 1; i <= 4; i++) {
			for (size_t j = 0; j < MAX_K; j++) {
				target_id[i][j] = index_table[base_id[i]][j];
				//printf("%d ", target_id[i][j]);
			}
		}

		int X1 = kp_base[base_id[1]].pt.x;
		int X2 = kp_base[base_id[2]].pt.x;
		int X3 = kp_base[base_id[3]].pt.x;
		int X4 = kp_base[base_id[4]].pt.x;

		int Y1 = kp_base[base_id[1]].pt.y;
		int Y2 = kp_base[base_id[2]].pt.y;
		int Y3 = kp_base[base_id[3]].pt.y;
		int Y4 = kp_base[base_id[4]].pt.y;
		int x1, x2, x3, x4, y1, y2, y3, y4;

		for (int i = 0; i < MAX_K; ++i) {
			x1 = kp_target[target_id[1][i]].pt.x;
			y1 = kp_target[target_id[1][i]].pt.y;
			for (int j = 0; j < MAX_K; ++j) {
				x2 = kp_target[target_id[2][j]].pt.x;
				y2 = kp_target[target_id[2][j]].pt.y;
				for (int k = 0; k < MAX_K; ++k) {
					x3 = kp_target[target_id[3][k]].pt.x;
					y3 = kp_target[target_id[3][k]].pt.y;
					for (int h = 0; h < MAX_K; ++h) {
						x4 = kp_target[target_id[4][h]].pt.x;
						y4 = kp_target[target_id[4][h]].pt.y;

						Mat H = construct_mat(X1, X2, X3, X4, Y1, Y2, Y3, Y4, x1, x2, x3, x4, y1, y2, y3, y4);
						int inlier_cnt = 0;
						for (int n = 0; n < kp_base.size(); ++n) {
							int nei = rand() % MAX_K;
							int tx = kp_target[index_table[n][nei]].pt.x;
							int ty = kp_target[index_table[n][nei]].pt.y;
							int bx = kp_base[n].pt.x;
							int by = kp_base[n].pt.y;

							float kp_data[] = { bx, by, 1.0 };
							Mat TA = H * Mat(3, 1, CV_32FC1, kp_data);

							int tx_ = TA.at<float>(0, 0) / TA.at<float>(2, 0);
							int ty_ = TA.at<float>(1, 0) / TA.at<float>(2, 0);


							float dis = sqrt((tx_ - tx)*(tx_ - tx) + (ty_ - ty)*(ty_ - ty));
							//printf("dis = %f\n", dis);
							if (dis < 10) inlier_cnt++;
						}
						if (inlier_cnt > max_inlier) {
							max_inlier = inlier_cnt;
							printf("inlier_cnt = %d\n", inlier_cnt);
							_best_affine = H;
						}
					}
				}
			}
		}
		if (max_inlier > (int)(0.28*kp_base.size())) {
			printMat(_best_affine);
			return;
		}
	}
}



bool isblack(Vec3b pixel) {
	if (pixel[0] == pixel[1] && pixel[2] == 0 && pixel[0] == pixel[2]) return true;
	return false;
}

void forward_warping(Mat img_1, Mat& img_2, Mat H) {
	//Mat Trans = homogrphy_sample_to_target * H;
	Mat Trans = H;
	for (size_t i = 0; i < img_1.rows; i++) {
		for (size_t j = 0; j < img_1.cols; j++) {
			float data[] = { j, i, 1 };
			Mat Y = Trans * Mat(3, 1, CV_32F, data);
			Y = Y / Y.at<float>(2, 0);
			if (isblack(img_1.at<Vec3b>(i, j))) continue;
			int fx = Y.at<float>(0, 0);
			int fy = Y.at<float>(1, 0);
			if (fx < 0) fx = 0;
			if (fy < 0) fy = 0;
			if (fy >= img_2.rows) fy = img_2.rows - 1;
			if (fx >= img_2.cols) fx = img_2.cols - 1;
			img_2.at<Vec3b>(fy, fx) = img_1.at<Vec3b>(i, j);
		}
	}
}

void backward_warping(Mat img_1, Mat& img_2, Mat H ){
	//Mat Trans = homogrphy_sample_to_target * H;
	//Trans = Trans.inv();
	Mat Trans = H.inv();
	for (int i = 0; i < img_2.rows; i++){
		for (int j = 0; j < img_2.cols; j++){
			float data[] = { j, i, 1 };
			Mat Y = Trans * Mat(3, 1, CV_32F, data);
			Y = Y / Y.at<float>(2, 0);

			int fx = Y.at<float>(0, 0);
			int fy = Y.at<float>(1, 0);

			if (0 <= fy && fy < img_1.rows && 0 <= fx && fx < img_1.cols){
				if (img_1.at<Vec3b>(fy, fx)[0] != 0 && img_1.at<Vec3b>(fy, fx)[1] != 0 && img_1.at<Vec3b>(fy, fx)[2] != 0) {

					img_2.at<Vec3b>(i, j)[0] = img_1.at<Vec3b>(fy, fx)[0]; //g
					img_2.at<Vec3b>(i, j)[1] = img_1.at<Vec3b>(fy, fx)[1]; //r
					img_2.at<Vec3b>(i, j)[2] = img_1.at<Vec3b>(fy, fx)[2]; //b
				}
			}
		}
	}
}

int main() {

	srand(time(NULL));
	// load image
	if (load_image()) printf("Load Image Successful.\n");
	else {
		printf("Load Image failed.\n");
		return -1;
	}

	// get homogrphy_sample_to_target
	cal_sift(keypoint_sample, keypoint_target, image_sample, image_target, descriptor_sample, descriptor_target);
	knn(descriptor_sample, descriptor_target);
	ransac(keypoint_sample, keypoint_target, homogrphy_sample_to_target);
	Mat test = Mat(image_target.rows+1000, image_target.cols+1000, CV_8UC3);
	for (int i = 0; i < pic_num; ++i) {
		printf("pic: %d\n", i + 1);
		cal_sift(keypoints[i], keypoint_sample, image_puzzles[i], image_sample, descriptor_puzzles[i], descriptor_sample);
		knn(descriptor_puzzles[i], descriptor_sample);
		ransac(keypoints[i], keypoint_sample, homogrphy[i]);
		//forward_warping(image_puzzles[i], test, homogrphy[i]);
		backward_warping(image_puzzles[i], test, homogrphy[i]);
		imshow("test", test);
		waitKey(0);
	}

	return 0;
}


bool knn(Mat base, Mat target) {

	for (int i = 0; i < base.rows; ++i) {
		index_table[i].clear();
		std::vector< std::pair<int, float> > dis;
		auto row_i = base.row(i);
		for (int j = 0; j < target.rows; ++j) {
			auto row_j = target.row(j);

			auto val = norm(row_i - row_j, NORM_L2);
			dis.push_back(std::make_pair(j, val));
		}
		std::sort(dis.begin(), dis.end(), cmp);
		for (int k = 0; k < MAX_K; ++k) {
			index_table[i].push_back(dis[k].first);
		}
	}

	return true;
}


bool load_image() {
	int obj;
	printf("Please choose your object: (1:logo, 2:table, 3:others) ");
	scanf("%d", &obj);
	if (obj == 1) {
		image_name = "logo";
	}
	else if (obj == 2) {
		image_name = "table";
	}
	else if (obj == 3) {
		printf("Input your object name: ");
		std::cin >> image_name;
	}
	else {
		return false;
	}
	std::string path_name = "test/" + image_name + "/";
	image_sample = imread(path_name + "sample.bmp");
	image_target = imread(path_name + "target.bmp");

	pic_num = 0;
	while (true) {
		std::string check_path = path_name + "puzzle" + std::to_string(pic_num + 1) + ".bmp";
		if (_access(check_path.c_str(), 0) != -1) pic_num++;
		else break;
	}
	printf("pic_num = %d\n", pic_num);
	for (int i = 0; i < pic_num; ++i) {
		image_puzzles.push_back(imread(path_name + "puzzle" + std::to_string(i + 1) + ".bmp"));
		//imshow("puzzle", image_puzzles[i]);
		//waitKey(0);
	}

	return true;
}


bool cal_sift(std::vector < KeyPoint>& keypoints_1, std::vector < KeyPoint>& keypoints_2, Mat& img_1, Mat& img_2, Mat& descriptor_1, Mat& descriptor_2) {
	// source image

	if (!img_1.data || !img_2.data) {
		std::cout << ".." << std::endl; return false;
	}

	//SIFT feature detector and feature extractor
	Ptr < xfeatures2d::SIFT>detectorSIFT = xfeatures2d::SIFT::create(2500);
	Ptr < xfeatures2d::SIFT>extractorSIFT = xfeatures2d::SIFT::create();

	detectorSIFT->detect(img_1, keypoints_1);
	detectorSIFT->detect(img_2, keypoints_2);

	Mat img_1_keypoints, img_2_keypoints;
	drawKeypoints(img_1, keypoints_1, img_1_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img_2, keypoints_2, img_2_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//imshow("INPUT_IMG", img_1);
	//imshow("SIFT_IMG", img_1_keypoints);
	//imshow("SIFT_IMG2", img_2_keypoints);
	imwrite("SIFT_IMG.bmp", img_1_keypoints);
	imwrite("SIFT_IMG2.bmp", img_2_keypoints);
	int key1 = (int)keypoints_1.size();
	int key2 = (int)keypoints_2.size();
	printf("Keypoint1=%d \nKeypoint2=%d\n", key1, key2);

	// Feature descriptor computation
	extractorSIFT->compute(img_1, keypoints_1, descriptor_1);
	extractorSIFT->compute(img_2, keypoints_2, descriptor_2);

	printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)\n", descriptor_1.size().height, descriptor_1.size().width,
		descriptor_2.size().height, descriptor_2.size().width);

	//waitKey(0);
	return true;
}
