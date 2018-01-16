// 颜色直方图
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

using namespace std;

cv::Mat rgb_hist(string str)
{
    cv::Mat img = cv::imread(str);
    vector<cv::Mat> bgr_planes;
    cv::split(img, bgr_planes);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::Mat hist;
    vconcat(r_hist, g_hist, hist);
    vconcat(hist, b_hist, hist);
    
    return hist;       
}

cv::Mat hsv_hist(string str)
{
    cv::Mat img = cv::imread(str);
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
    vector<cv::Mat> hsv_planes;
    cv::split(img, hsv_planes);
    int h_bin = 181;
    int s_bin = 256;
    int v_bin = 256;
    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    float v_range[] = {0, 256};
    const float* h_ranges = {h_range};
    const float* s_ranges = {s_range};
    const float* v_ranges = {v_range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat h_hist, s_hist, v_hist;
    cv::calcHist( &hsv_planes[0], 1, 0, cv::Mat(), h_hist, 1, &h_bin, &h_ranges, uniform, accumulate );
    cv::calcHist( &hsv_planes[1], 1, 0, cv::Mat(), s_hist, 1, &s_bin, &s_ranges, uniform, accumulate );
    cv::calcHist( &hsv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &v_bin, &v_ranges, uniform, accumulate );
    cv::Mat hist;
    cv::vconcat(h_hist, s_hist, hist);
    cv::vconcat(hist, v_hist, hist);
    
    return hist;     
}

int main( int argc, char** argv )
{
  cv::Mat hist1 = rgb_hist(argv[1]);
  cv::Mat hist2 = rgb_hist(argv[2]);
  double corr = cv::compareHist(hist1, hist2, CV_COMP_CORREL);
  cout << corr << endl;
  return 0;
}
