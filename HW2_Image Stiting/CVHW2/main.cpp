#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <stdio.h>
#include <string.h>

using namespace cv;

int main() {

	// source image
	char* img_1_name = "./test/logo/sample.bmp";
	char* img_2_name = "./test/logo/target.bmp";

	// read image
	Mat img_1 = imread(img_1_name);
	Mat img_2 = imread(img_2_name);
	if (!img_1.data || !img_2.data) {
		std::cout << "鮫�颪�よみこめません" << std::endl; return -1;
	}

	//SIFT feature detector and feature extractor
	Ptr < xfeatures2d::SIFT>detectorSIFT = xfeatures2d::SIFT::create(0.05,5.0);
	Ptr < xfeatures2d::SIFT>extractorSIFT = xfeatures2d::SIFT::create(3.0);

	std::vector < KeyPoint>keypoints_1, keypoints_2;
	detectorSIFT->detect(img_1, keypoints_1);
	detectorSIFT->detect(img_2, keypoints_2);

	Mat img_1_keypoints, img_2_keypoints;
	drawKeypoints(img_1, keypoints_1, img_1_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img_2, keypoints_2, img_2_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//imshow("INPUT_IMG", img_1);
	imshow("SIFT_IMG", img_1_keypoints);
	imshow("SIFT_IMG2", img_2_keypoints);
	imwrite("SIFT_IMG.bmp", img_1_keypoints);
	imwrite("SIFT_IMG2.bmp", img_2_keypoints);
	int key1 = (int)keypoints_1.size();
	int key2 = (int)keypoints_2.size();
	printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);

	// Feature descriptor computation
	Mat descriptor_1, descriptor_2;
	extractorSIFT->compute(img_1, keypoints_1, descriptor_1);
	extractorSIFT->compute(img_2, keypoints_2, descriptor_2);

	printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)", descriptor_1.size().height, descriptor_1.size().width,
		descriptor_2.size().height, descriptor_2.size().width);

	waitKey(0);
	return 0;
}

