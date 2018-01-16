// 参考 http://blog.csdn.net/nnnnnnnnnnnny/article/details/52182091
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

int main(int argc, char** argv)
{
	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
	//cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create();
	//cv::Ptr<cv::Feature2D> f2d = cv::ORB::create();
	
	cv::Mat img_1 = cv::imread("1.png", 0);
	cv::Mat img_2 = cv::imread("2.png", 0);

	vector<cv::KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_1, keypoints_1);
	f2d->detect(img_2, keypoints_2);
	
	cv::Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);

    cv::Mat sift_results;                    
    cv::drawKeypoints(img_1, keypoints_1, sift_results);
    cv::imshow("sift_result", sift_results);  

	cv::BFMatcher matcher;
	vector<cv::DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	cv::Mat img_matches;
	cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	cv::imshow("matched result", img_matches);
	cv::waitKey();

	return 0;
}
