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
#define THRESHOLD_GOODRESULT	10					    // distance less this threshold would consider as good match result
#define MAX_POIONT              2048

std::string image_name;
std::vector < KeyPoint>keypoints[MAX_PIC_SIZE];
std::vector < KeyPoint> keypoint_sample;
std::vector < KeyPoint> keypoint_target;
std::vector < int > index_table[MAX_POIONT];
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

bool cmp(std::pair<int, float> x, std::pair<int, float> y) {
	return x.second < y.second;
}

bool load_image();
bool cal_sift(std::vector < KeyPoint>& keypoints_1, std::vector < KeyPoint>& keypoints_2, Mat& img_1, Mat& img_2, Mat& descriptor_1, Mat& descriptor_2);
bool knn(Mat base, Mat target);
void printde() {std::cout << "debug" << std::endl;}
void printMat(Mat M) {
	for (size_t i = 0; i < M.rows; i++) {
		for (size_t j = 0; j < M.cols; j++) {
			std::cout << M.at<float>(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void ransac( std::vector < KeyPoint>& kp_base, std::vector<KeyPoint>& kp_target, Mat& _best_affine) {

	int max_cnt = 0;
	for (int r = 0; r < RANSAC_ROUND; ++r) {

		float x[5], y[5], u[5][5], v[5][5];
		int hash[MAX_POIONT] = { 0 }, id;

		for (int i = 0; i < 4; i++) {
			id = rand() % kp_base.size();
			while (hash[id]) {
				id = rand() % kp_base.size();
			}
			hash[id] = 1;
			x[i] = kp_base[id].pt.x;
			y[i] = kp_base[id].pt.y;
		
			for (int j = 0; j < MAX_K; j++) {
				u[i][j] = kp_target[index_table[id][j]].pt.x;
				v[i][j] = kp_target[index_table[id][j]].pt.y;
			}
		}

		for (int i = 0; i < MAX_K; ++i) {
			int x1 = u[0][i];
			int y1 = v[0][i];
			for (int j = 0; j < MAX_K; ++j) {
				int x2 = u[1][j];
				int y2 = v[1][j];
				for (int k = 0; k < MAX_K; ++k) {
					int x3 = u[2][k];
					int y3 = v[2][k];
					for (int jj = 0; jj < MAX_K; ++jj) {
						int x4 = u[3][jj];
						int y4 = v[3][jj];
						
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
						int cnt = 0;
						for (int kp = 0; kp <  kp_base.size(); ++kp) {
							int xx = kp_target[ index_table[kp][0]].pt.x;
							int yy = kp_target[index_table[kp][0]].pt.y;
								
							float kp_data[] = { kp_base[kp].pt.x, kp_base[kp].pt.y, 1.0 };
							Mat Y = H * Mat(3, 1, CV_32FC1, kp_data);

							int x_ = Y.at<float>(0, 0) / Y.at<float>(2, 0);
							int y_ = Y.at<float>(1, 0) / Y.at<float>(2, 0);

							float dis = sqrt((x_ - xx)*(x_ - xx)+ (y_ - yy)*(y_ - yy));
							//printf("dis = %f\n", dis);
							
							if (dis < 50) cnt++;
						}
						if (cnt > max_cnt) {
							max_cnt = cnt;
							printf("cnt = %d\n", cnt);
							_best_affine = H;
						}
					}
				}
			}
		}
		if (max_cnt > (int)(0.25*kp_base.size())) {
			printMat(_best_affine);
			return;
		}
	}
}



bool isblack(Vec3b pixel) {
	if (pixel[0] == pixel[1] && pixel[2] == 0 && pixel[0] == pixel[2]) return true;
	return false;
}

void backwardWarping(Mat img_1, Mat trans, Mat& img_2)
{
	printMat(trans);
	printMat(homogrphy_sample_to_target);
	Mat homo = homogrphy_sample_to_target * trans;
	homo = homo.inv();

	for (int i = 0; i < img_2.rows; i++)
	{
		for (int j = 0; j < img_2.cols; j++)
		{
			float data[] = { j, i, 1 };
			Mat B(3, 1, CV_32F, data );
			Mat A = homo * B;
			A = A / A.at<float>(2, 0);
			if (0 <= A.at<float>(1, 0) && A.at<float>(1, 0) < img_1.rows && 0 <= A.at<float>(0, 0) && A.at<float>(0, 0) < img_1.cols)
			{
				if (img_1.at<Vec3b>((int)A.at<float>(1, 0), (int)A.at<float>(0, 0))[0] != 0 && img_1.at<Vec3b>((int)A.at<float>(1, 0), (int)A.at<float>(0, 0))[1] != 0 && img_1.at<Vec3b>((int)A.at<float>(1, 0), (int)A.at<float>(0, 0))[2] != 0) {
					img_2.at<Vec3b>(i, j)[0] = img_1.at<Vec3b>((int)A.at<float>(1, 0), (int)A.at<float>(0, 0))[0];//g
					img_2.at<Vec3b>(i, j)[1] = img_1.at<Vec3b>((int)A.at<float>(1, 0), (int)A.at<float>(0, 0))[1];//r
					img_2.at<Vec3b>(i, j)[2] = img_1.at<Vec3b>((int)A.at<float>(1, 0), (int)A.at<float>(0, 0))[2];//b
				}
			}
		}
	}
}
using namespace std;
Mat affineMappingMatrix(Mat puzzleDescriptor, vector<KeyPoint> puzzleKeypoints, vector<KeyPoint> sampleKeypoints)
{
	printf("---------------------This is another round to find the best Homography matrix---------------------\n");
	int k = 3, kk, nn, inlier = 0, Th = 1, max_inlier = 0;
	float diffVector1, diffVector2, totalDiff, sqrtTotalDiff;
	vector<int> point(3);
	Mat A(6, 6, CV_32F);
	Mat U(6, 1, CV_32F);
	Mat best_H(3, 3, CV_32F);
	Mat H(3, 3, CV_32F);
	Mat hOther(3, 1, CV_32F);
	Mat A2(3, 1, CV_32F);
	Mat A2Sure(3, 1, CV_32F);
	for (int t = 0; t < 5000; t++)
	{
		do
		{
			point[0] = rand() % puzzleDescriptor.rows;
			point[1] = rand() % puzzleDescriptor.rows;
			point[2] = rand() % puzzleDescriptor.rows;

		} while ((puzzleKeypoints[point[0]].pt.x == 0 && puzzleKeypoints[point[0]].pt.y == 0) ||
			(puzzleKeypoints[point[1]].pt.x == 0 && puzzleKeypoints[point[1]].pt.y == 0) ||
			(puzzleKeypoints[point[2]].pt.x == 0 && puzzleKeypoints[point[2]].pt.y == 0));

		for (int r = 0, m = 0; r < k, m < k; r = r + 2, m++)
		{
			A.at<float>(r, 0) = puzzleKeypoints[point[m]].pt.x;
			A.at<float>(r, 1) = puzzleKeypoints[point[m]].pt.y;
			A.at<float>(r, 2) = 1;
			A.at<float>(r, 3) = 0;
			A.at<float>(r, 4) = 0;
			A.at<float>(r, 5) = 0;
			A.at<float>(r + 1, 0) = 0;
			A.at<float>(r + 1, 1) = 0;
			A.at<float>(r + 1, 2) = 0;
			A.at<float>(r + 1, 3) = puzzleKeypoints[point[m]].pt.x;
			A.at<float>(r + 1, 4) = puzzleKeypoints[point[m]].pt.y;
			A.at<float>(r + 1, 5) = 1;
			nn = rand() % 3;
			switch (nn)
			{
			case 0:
				kk = 0;
				break;
			case 1:
				kk = 1;
				break;
			case 2:
				kk = 2;
				break;
			}
			U.at<float>(r, 0) = sampleKeypoints[ index_table[point[m]][kk]].pt.x;
			U.at<float>(r + 1, 0) = sampleKeypoints[index_table[point[m]][kk]].pt.y;
		}
		Mat ATRAN = A.t();

		Mat H1 = ((ATRAN*A).inv())*ATRAN*U;
		H.at<float>(0, 0) = H1.at<float>(0, 0);
		H.at<float>(0, 1) = H1.at<float>(1, 0);
		H.at<float>(0, 2) = H1.at<float>(2, 0);
		H.at<float>(1, 0) = H1.at<float>(3, 0);
		H.at<float>(1, 1) = H1.at<float>(4, 0);
		H.at<float>(1, 2) = H1.at<float>(5, 0);
		H.at<float>(2, 0) = 0;
		H.at<float>(2, 1) = 0;
		H.at<float>(2, 2) = 1;
		inlier = 0;
		for (int i = 0; i < puzzleDescriptor.rows; i++)
		{
			if (puzzleKeypoints[i].pt.x != 0 && puzzleKeypoints[i].pt.y != 0)
			{
				hOther.at<float>(0, 0) = puzzleKeypoints[i].pt.x;
				hOther.at<float>(1, 0) = puzzleKeypoints[i].pt.y;
				hOther.at<float>(2, 0) = 1;
				A2 = H*hOther;
				/*This procedure is try to make sure A2(2,0) is definitely be 1*/
				A2Sure = A2 / A2.at<float>(2, 0);
				for (int j = 0; j < k; j++) {
					diffVector1 = A2Sure.at<float>(0, 0) - sampleKeypoints[index_table[i][j]].pt.x;
					diffVector1 = diffVector1*diffVector1;
					diffVector2 = A2Sure.at<float>(1, 0) - sampleKeypoints[index_table[i][j]].pt.y;
					diffVector2 = diffVector2*diffVector2;
					totalDiff = diffVector1 + diffVector2;
					sqrtTotalDiff = sqrt(totalDiff);
					if (sqrtTotalDiff < Th)
					{
						inlier = inlier + 1;
					}
				}
			}
		}
		if (inlier > max_inlier)
		{
			max_inlier = inlier;
			H.copyTo(best_H);
		}

	}
	return best_H;
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
	//ransac(keypoint_sample, keypoint_target, homogrphy_sample_to_target);
	homogrphy_sample_to_target = affineMappingMatrix(descriptor_sample, keypoint_sample, keypoint_target);
	for (int i = 0; i < pic_num; ++i) {
		printf("pic: %d\n", i + 1);
		cal_sift(keypoints[i], keypoint_sample, image_puzzles[i], image_sample, descriptor_puzzles[i], descriptor_sample);
		knn(descriptor_puzzles[i], descriptor_sample);
		//ransac(keypoints[i], keypoint_sample, homogrphy[i]);
		homogrphy[i] = affineMappingMatrix(descriptor_puzzles[i], keypoints[i], keypoint_sample);
		backwardWarping(image_puzzles[i], homogrphy[i], image_target);
		imshow("test", image_target);
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
	printf("K = 3 : %d %d %d\n", index_table[0][0], index_table[0][1], index_table[0][2]);
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

//void ransac(std::vector < pair_node >& knn_pair, std::vector < KeyPoint>& kp_base, std::vector<KeyPoint>& kp_target, Mat& _best_affine) {
//
//	int max_cnt = 0;
//	Mat H(3, 3, CV_32F, Scalar(0));
//	for (int r = 0; r < 5000; ++r) {
//
//		int index[4] = { 0 };
//		for (size_t i = 0; i < 4; i++){
//			index[i] = rand() % knn_pair.size();
//			for (size_t j = 0; j < 4; j++){
//				if ( (i !=j ) && index[i] == index[j]){
//					index[i] = rand() % knn_pair.size();
//					j = 0;
//				}
//			}
//		}
//
//		int X1 = kp_base[knn_pair[index[0]].idx_base].pt.x;
//		int Y1 = kp_base[knn_pair[index[0]].idx_base].pt.y;
//		int X2 = kp_base[knn_pair[index[1]].idx_base].pt.x;
//		int Y2 = kp_base[knn_pair[index[1]].idx_base].pt.y;
//		int X3 = kp_base[knn_pair[index[2]].idx_base].pt.x;
//		int Y3 = kp_base[knn_pair[index[2]].idx_base].pt.y;
//		int X4 = kp_base[knn_pair[index[3]].idx_base].pt.x;
//		int Y4 = kp_base[knn_pair[index[3]].idx_base].pt.y;
//
//		int x1 = kp_target[knn_pair[index[0]].idx_target].pt.x;
//		int y1 = kp_target[knn_pair[index[0]].idx_target].pt.y;
//		int x2 = kp_target[knn_pair[index[1]].idx_target].pt.x;
//		int y2 = kp_target[knn_pair[index[1]].idx_target].pt.y;
//		int x3 = kp_target[knn_pair[index[2]].idx_target].pt.x;
//		int y3 = kp_target[knn_pair[index[2]].idx_target].pt.y;
//		int x4 = kp_target[knn_pair[index[3]].idx_target].pt.x;
//		int y4 = kp_target[knn_pair[index[3]].idx_target].pt.y;
//
//		Mat A(8, 9, CV_32F, Scalar(0));
//
//		float data1[] = { X1, Y1, 1, 0, 0, 0, -x1*X1, -x1*Y1, -x1 };
//		Mat(1, 9, CV_32FC1, data1).copyTo(A.row(0));
//		float data2[] = { 0, 0, 0, X1, Y1, 1, -y1*X1, -y1*Y1, -y1 };
//		Mat(1, 9, CV_32FC1, data2).copyTo(A.row(1));
//		float data3[] = { X2, Y2, 1, 0, 0, 0, -x2*X2, -x2*Y2, -x2 };
//		Mat(1, 9, CV_32FC1, data3).copyTo(A.row(2));
//		float data4[] = { 0, 0, 0, X2, Y2, 1, -y2*X2, -y2*Y2, -y2 };
//		Mat(1, 9, CV_32FC1, data4).copyTo(A.row(3));
//		float data5[] = { X3, Y3, 1, 0, 0, 0, -x3*X3, -x3*Y3, -x3 };
//		Mat(1, 9, CV_32FC1, data5).copyTo(A.row(4));
//		float data6[] = { 0, 0, 0, X3, Y3, 1, -y3*X3, -y3*Y3, -y3 };
//		Mat(1, 9, CV_32FC1, data6).copyTo(A.row(5));
//		float data7[] = { X4, Y4, 1, 0, 0, 0, -x4*X4, -x4*Y4, -x4 };
//		Mat(1, 9, CV_32FC1, data7).copyTo(A.row(6));
//		float data8[] = { 0, 0, 0, X4, Y4, 1, -y4*X4, -y4*Y4, -y4 };
//		Mat(1, 9, CV_32FC1, data8).copyTo(A.row(7));
//
//		//printMat(A);
//
//
//		Mat AmatrixEigenValue, AmatrixEigenVector;
//		eigen(A.t()*A, AmatrixEigenValue, AmatrixEigenVector);
//
//		Mat H = AmatrixEigenVector.row(8);
//		H = H.reshape(0, 3);
//
//		int cnt = 0;
//		int size = kp_base.size();
//		//printf("sz = %d\n", size);
//		for (int kp = 0; kp < size; ++kp) {
//			float xx = kp_target[knn_pair[kp].idx_target].pt.x;
//			float yy = kp_target[knn_pair[kp].idx_target].pt.y;
//
//			float kp_data[] = { kp_base[knn_pair[kp].idx_base].pt.x, kp_base[knn_pair[kp].idx_base].pt.y, 1.0 };
//			Mat Y = H*Mat(3, 1, CV_32FC1, kp_data);
//
//			float x_ = Y.at<float>(0, 0) / Y.at<float>(2, 0);
//			float y_ = Y.at<float>(1, 0) / Y.at<float>(2, 0);
//
//			float dis = (x_ - xx)*(x_ - xx) + (y_ - yy)*(y_ - yy);
//			//printf("dis = %f\n", dis);
//			if (dis < 10) {
//				cnt++;
//			}
//		}
//		if (cnt > max_cnt) {
//			max_cnt = cnt;
//			printf("cnt = %d\n", cnt);
//			_best_affine = H;
//		}
//		if (max_cnt > (int)(0.5*kp_base.size())) {
//			_best_affine = H;
//			//printf("dis = %f\n", dis);
//			printf("cnt = %d\n", cnt);
//			printMat(H);
//			return;
//		}
//	}
//}