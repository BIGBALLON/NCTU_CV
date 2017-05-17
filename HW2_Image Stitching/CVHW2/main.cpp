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
#define MAX_K                   3
#define RANSAC_ROUND			1000				    // round of RANSAC
#define NUM_REQ_HOMOGRAPHY		4					    // required points to build a homography
#define HOMOGRAPHY_H_HEIGHT    (NUM_REQ_HOMOGRAPHY<<1)	// height of H when build homography
#define HOMOGRAPHY_H_WIDTH		9					    // width of H when build homography
#define THRESHOLD_GOODRESULT	5					    // distance less this threshold would consider as good match result

std::string image_name;
std::vector < KeyPoint>keypoints[MAX_PIC_SIZE];
std::vector < KeyPoint> keypoint_sample;
std::vector < KeyPoint> keypoint_target;
int pic_num;

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

// knn pair

struct pair_node {
	const int idx;
	const int idx_base;
	const int idx_target;

	pair_node():idx(-1), idx_target(-1), idx_base(-1) {}
	pair_node(int a, int b, int c) :idx(a), idx_base(b), idx_target(c) {}

	static int get_base_idx(pair_node p) { return p.idx_base; }
	static int get_target_idx(pair_node p) { return p.idx_target; }
};

bool cmp(std::pair<int, float> x, std::pair<int, float> y) {
	return x.second < y.second;
}

bool load_image();
bool cal_sift(std::vector < KeyPoint>& keypoints_1, std::vector < KeyPoint>& keypoints_2, Mat& img_1, Mat& img_2, Mat& descriptor_1, Mat& descriptor_2);
bool knn(Mat& base, Mat& target, std::vector < pair_node >& knn_pair);
void printde() {
	std::cout << "debug" << std::endl;
}
void printMat(Mat M) {
	for (size_t i = 0; i < M.rows; i++)
	{
		for (size_t j = 0; j < M.cols; j++)
		{
			std::cout << M.at<float>(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void ransac(std::vector < pair_node >& knn_pair, std::vector < KeyPoint>& kp_base, std::vector<KeyPoint>& kp_target, Mat& _best_affine) {

	int max_cnt = 0;
	for (int r = 0; r < RANSAC_ROUND; ++r) {
		
		float x[5], y[5], u[5][5], v[5][5], id;
		for (size_t i = 0; i < 4; i++){
			id = rand() % kp_base.size();
			x[i] = kp_base[id].pt.x;
			y[i] = kp_base[id].pt.y;
			 
			int target_i = id*MAX_K;
			for (size_t j = 0; j < MAX_K; j++){
				u[i][j] = kp_target[knn_pair[target_i + j].idx_target].pt.x;
				v[i][j] = kp_target[knn_pair[target_i + j].idx_target].pt.y;
			}
			
		}

		for (int i = 0; i < MAX_K; ++i) {
			int x1 = u[0][i];
			int y1 = v[0][i];
			for (int j = 0; j < MAX_K; ++j) {
				int x2 = u[1][j];
				int y2 = v[1][j];
				for (int ii = 0; ii < MAX_K; ++ii) {
					int x3 = u[2][ii];
					int y3 = v[2][ii];
					for (int jj = 0; jj < MAX_K; ++jj) {
						int x4 = u[3][jj];
						int y4 = v[3][jj];
						/*
						Mat A = Mat(8, 9, CV_32FC1);
						
						float data1[] = { x[0], y[0], 1, 0, 0, 0, -x1*x[0], -x1*y[0], -x1 };
						Mat(1, 9, CV_32FC1, data1).copyTo(A.row(0));
						float data2[] = { 0, 0, 0, x[0], y[0], 1, -y1*x[0], -y1*y[0], -y1 };
						Mat(1, 9, CV_32FC1, data2).copyTo(A.row(1));
						float data3[] = { x[1], y[1], 1, 0, 0, 0, -x2*x[1], -x2*y[1], -x2 };
						Mat(1, 9, CV_32FC1, data3).copyTo(A.row(2));
						float data4[] = { 0, 0, 0, x[1], y[1], 1, -y2*y[1], -y2*y[1], -y2 };
						Mat(1, 9, CV_32FC1, data4).copyTo(A.row(3));
						float data5[] = { x[2], y[2], 1, 0, 0, 0, -x3*x[2], -x3*y[2], -x3 };
						Mat(1, 9, CV_32FC1, data5).copyTo(A.row(4));
						float data6[] = { 0, 0, 0, x[2], y[2], 1, -y3*x[2], -y3*y[2], -y3 };
						Mat(1, 9, CV_32FC1, data6).copyTo(A.row(5));
						float data7[] = { x[3], y[3], 1, 0, 0, 0, -x4*x[3], -x4*y[3], -x4 };
						Mat(1, 9, CV_32FC1, data7).copyTo(A.row(6));
						float data8[] = { 0, 0, 0, x[3], y[3], 1, -y4*x[3], -y4*y[3], -y4 };
						Mat(1, 9, CV_32FC1, data8).copyTo(A.row(7));



						
						Mat AmatrixEigenValue = Mat::zeros(8, 9, CV_32FC1);
						Mat AmatrixEigenVector = Mat::zeros(8, 9, CV_32FC1);
						auto AmatrixTAmatrix = A.t()*A;
						
						eigen(AmatrixTAmatrix, AmatrixEigenValue, AmatrixEigenVector);
						
						Mat H = AmatrixEigenVector.row(8);
						H = H.reshape(0, 3);
						*/
						float data1[] = { x1, x2, x3, y1, y2, y3 };
						Mat A = Mat(2, 3, CV_32FC1, data1);
						float data2[] = { x[0], x[1], x[2], y[0], y[1], y[2], 1, 1, 1 };
						Mat B = Mat(3, 3, CV_32FC1, data2);
						Mat H = A * B.inv();

						int cnt = 0;
						int size = kp_base.size();
						//printf("sz = %d\n", size);
						for (int kp = 0; kp < size; ++kp) {
							int target_i = kp*MAX_K;
							float xx = kp_target[knn_pair[target_i].idx_target].pt.x;
							float yy = kp_target[knn_pair[target_i].idx_target].pt.y;

							float kp_data[] = { kp_base[kp].pt.x, kp_base[kp].pt.y, 1.0 };
							Mat Y = H*Mat(3, 1, CV_32FC1, kp_data);
							
							/*Y.at<float>(0, 0) = Y.at<float>(0, 0) / Y.at<float>(2, 0);
							Y.at<float>(1, 0) = Y.at<float>(1, 0) / Y.at<float>(2, 0);
							Y.at<float>(2, 0) = Y.at<float>(2, 0) / Y.at<float>(2, 0);
*/
							float dis = sqrt((Y.at<float>(0,0) - xx)*(Y.at<float>(0,0) - xx) + (Y.at<float>(1,0) - yy)*(Y.at<float>(1,0) - yy));
							//printf("dis = %f\n", dis);
							if (dis < 20) {
								cnt++;
							}
						}
						if (cnt > max_cnt) {
							max_cnt = cnt;
							printf("cnt = %d\n", cnt);
							printMat(H);
							_best_affine = H;
						}
						if (max_cnt > (int)(0.28*kp_base.size()) ) {
							_best_affine = H;
							//printf("dis = %f\n", dis);
							printf("cnt = %d\n", cnt);
							printMat(H);
							return;
						}
					}
				}
			}
		}
/*
		for (int i = 0; i < 4; ++i) {
			id = rand() % kp_base.size();
			x[i] = kp_base[id].pt.x;
			y[i] = kp_base[id].pt.y;
			int target_i = knn_pair[id].idx_target;
			u[i] = kp_target[target_i].pt.x;
			v[i] = kp_target[target_i].pt.y;
			printf("%f, %f, %f, %f\n", x[i], y[i], u[i], v[i]);
		}

		Mat A = Mat(8, 9, CV_32FC1);
		for (int n = 0; n < 4; ++n) {
			for (int i = 0; i < 8; ++i) {
				
				if (i == 0 && n == 0 ) {
					float tmp[9] = { x[n], y[n], 1, 0, 0, 0, -u[n] * x[n], -u[n] * y[n], -u[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}
				else if (i == 1 && n == 0) {
					float tmp[9] = { 0, 0, 0, x[n], y[n], 1, -v[n] * x[n], -v[n] * y[n], -v[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}else if (i == 2 && n == 1 ) {
					float tmp[9] = { x[n], y[n], 1, 0, 0, 0, -u[n] * x[n], -u[n] * y[n], -u[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}
				else if (i == 3 && n == 1) {
					float tmp[9] = { 0, 0, 0, x[n], y[n], 1, -v[n] * x[n], -v[n] * y[n], -v[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}else if (i == 4 && n == 2) {
					float tmp[9] = { x[n], y[n], 1, 0, 0, 0, -u[n] * x[n], -u[n] * y[n], -u[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}
				else if (i == 5 && n == 2) {
					float tmp[9] = { 0, 0, 0, x[n], y[n], 1, -v[n] * x[n], -v[n] * y[n], -v[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}
				else if (i == 6 && n == 3) {
					float tmp[9] = { x[n], y[n], 1, 0, 0, 0, -u[n] * x[n], -u[n] * y[n], -u[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}else if (i == 7 && n == 3) {
					float tmp[9] = { 0, 0, 0, x[n], y[n], 1, -v[n] * x[n], -v[n] * y[n], -v[n] };
					Mat t = Mat(1, 9, CV_32FC1, tmp);
					t.copyTo(A.row(i));
				}
			}
		}
		*/
		/*
		Mat AmatrixEigenValue = Mat::zeros(8, 9, CV_32FC1);
		Mat AmatrixEigenVector = Mat::zeros(8, 9, CV_32FC1);
		auto AmatrixTAmatrix = A.t()*A;
		
		eigen(AmatrixTAmatrix, AmatrixEigenValue, AmatrixEigenVector);
		
		
		Mat H = AmatrixEigenVector.row(8);
		H = H.reshape(0, 3);
		_best_affine = H;
		*/
	}
}


bool isblack(Vec3b pixel) {
	if (pixel[0] == pixel[1] && pixel[2] == 0 && pixel[0] == pixel[2]) return true;
	return false;
}

int main() {

	srand(time(NULL));
	// load image
	if (load_image()) printf("Load Image Successful.\n");
	else {
		printf("Load Image failed.\n");
		return -1;
	}

	Mat train[MAX_PIC_SIZE], train2;
	for (int i = 0; i < pic_num; ++i) {

		cal_sift(keypoints[i], keypoint_sample, image_puzzles[i], image_sample, descriptor_puzzles[i], descriptor_sample);
		
		std::vector < pair_node > cur_knn_pair;
		knn(descriptor_puzzles[i], descriptor_sample, cur_knn_pair);
		ransac(cur_knn_pair, keypoints[i], keypoint_sample, train[i]);
	}

	cal_sift(keypoint_sample, keypoint_target, image_sample, image_target, descriptor_sample, descriptor_target);
	std::vector < pair_node > cur_knn_pair;
	knn(descriptor_sample, descriptor_target, cur_knn_pair);
	
	ransac(cur_knn_pair, keypoint_sample, keypoint_target, train2);

	//Mat result(image_target);

	Mat result = Mat(image_sample.rows, image_sample.cols, CV_8UC3);
	for (size_t n = 0; n < pic_num; n++)
	{
		for (size_t i = 0; i < image_puzzles[n].rows; i++)
		{
			for (size_t j = 0; j < image_puzzles[n].cols; j++)
			{
	
				float tmp[3] = { i, j, 1.0 };
				Mat X = Mat(3, 1, CV_32F, tmp );
				if (isblack(image_puzzles[n].at<Vec3b>(i, j))) {
					continue;
				}

				//printMat(X);
				Mat x = train[n] * X;
				//Mat x = train2 * train[n] * X;
/*
				x.at<float>(0,0) = x.at<float>(0,0) / x.at<float>(2, 0);
				x.at<float>(1,0) = x.at<float>(1,0) / x.at<float>(2, 0);
				x.at<float>(2,0) = x.at<float>(2,0) / x.at<float>(2, 0);*/
			
				//printMat(x);

				if (x.at<float>(0) < 0 || x.at<float>(0) >= image_puzzles[n].cols || x.at<float>(1) < 0 || x.at<float>(1) >= image_puzzles[n].rows) {
					continue;
				}
				result.at<Vec3b>(x.row(0).at<float>(0), x.row(1).at<float>(0))[0] = image_puzzles[n].at<Vec3b>(i, j)[0];
				result.at<Vec3b>(x.row(0).at<float>(0), x.row(1).at<float>(0))[1] = image_puzzles[n].at<Vec3b>(i, j)[1];
				result.at<Vec3b>(x.row(0).at<float>(0), x.row(1).at<float>(0))[2] = image_puzzles[n].at<Vec3b>(i, j)[2];
			}
		}
	}

	imshow("remap result", result);
	waitKey(0);
	return 0;
}


bool knn(Mat& base, Mat& target, std::vector < pair_node >& knn_pair) {
	
	int pair_num = 0;
	for (int i = 0; i < base.rows; ++i) {
		std::vector< std::pair<int, float> > dis;
		auto row_i = base.row(i);
		for (int j = 0; j < target.rows; ++j) {
			auto row_j = target.row(j);

			auto val = norm(row_i - row_j, NORM_L2);

			dis.push_back(std::make_pair(j, val));
		}

		std::sort(dis.begin(), dis.end(), cmp);
		for (int k = 0; k < MAX_K; ++k) {
			knn_pair.push_back(pair_node(pair_num++, i, dis[k].first));
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
		if (_access(check_path.c_str(),0) != -1) pic_num++;
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
		std::cout << "鮫�颪�よみこめません" << std::endl; return false;
	}

	//SIFT feature detector and feature extractor
	Ptr < xfeatures2d::SIFT>detectorSIFT = xfeatures2d::SIFT::create();
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