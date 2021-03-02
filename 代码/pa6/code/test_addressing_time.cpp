#include <Eigen/Core>
#include <Eigen/Dense>
#include <bits/stdc++.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "../1.png"; // first image
string file_2 = "../2.png"; // second image

inline float GetPixelValue(const cv::Mat& img, float x, float y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}

int main()
{
    //addressing
    clock_t start, finish;
    const int routine_time = 1e8;
    volatile int a = 100000;
    start = clock();
    for (int i = 0; i < routine_time; ++i) {
        if (i < a) {
            a++;
        }
    }
    finish = clock();
    cout << finish - start << endl;
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);
    //GetPixelValue
    start = clock();
    for (int i = 0; i < routine_time; ++i) {
        a = GetPixelValue(img1, 1, 1);
    }
    finish = clock();
    cout << finish - start << endl;
    //get just addressing value
    Eigen::Vector2d vec;
    vec << 1, 1;
    Eigen::Matrix2d mat = Eigen::Matrix2d::Zero();
    start = clock();
    for (int i = 0; i < routine_time; ++i) {
        mat += vec * vec.transpose();
    }
    finish = clock();
    cout << mat << endl;
    cout << finish - start << endl;
}