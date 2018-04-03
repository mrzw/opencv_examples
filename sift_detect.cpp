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

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	vector<cv::DMatch> matches;
///////////////////////////////////////////////////////////
	//根据匹配点最小距离筛选匹配点
	matcher->match(descriptors_1, descriptors_2, matches);
	double min_dist=10000, max_dist=0;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
	}
	vector< cv::DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
	}
	//根据knn和ransac删除匹配点
	vector<vector<cv::DMatch>> matches_knn;
    matcher->knnMatch(descriptors_1, descriptors_2, matches_knn, 2);
    for(int i=0; i<matches_knn.size(); i++) {
        if(matches_knn[i][0].distance < 0.8*matches_knn[i][1].distance) {
            matches.push_back(matches_knn[i][0]);
        }
    }
    vector<cv::DMatch> inlierMatches;
    vector<cv::KeyPoint> leftKeypoints;
    vector<cv::KeyPoint> rightKeypoints;
    cv::Mat fundamental_matrix;
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;
    for(int i=0; i<matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    vector<uchar> status;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, status); // 通过基础矩阵删除外点，提高匹配正确率
    for(int i=0, index=0; i<matches.size(); i++) {
        if(status[i]!=0) {
            leftKeypoints.push_back(keypoints_1[matches[i].queryIdx]);
            rightKeypoints.push_back(keypoints_2[matches[i].trainIdx]);
            matches[i].queryIdx = index;
            matches[i].trainIdx = index;
            inlierMatches.push_back(matches[i]);
            index++;
        }
    }
/////////////////////////////////////////////////////////////////
	cv::Mat img_matches;
	cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	cv::imshow("matched result", img_matches);
	cv::waitKey();

	return 0;
}
