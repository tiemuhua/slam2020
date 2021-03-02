#include <Eigen/Core>
#include <boost/format.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace Sophus;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "../left.png";
string disparity_file = "../disparity.png";
boost::format fmt_others("../%06d.png"); // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const VecVector2d& px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d& T21);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const VecVector2d& px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d& T21);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat& img, float x, float y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}

int main(int argc, char** argv)
{
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000;
    int boarder = 40;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder); // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder); // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    vector<Mat> imgs(6);
    imgs[0] = left_img;
    for (int i = 1; i < 6; i++) { // 1~10
        imgs[i] = cv::imread((fmt_others % i).str(), 0);
        SE3d T_cur;
        DirectPoseEstimationMultiLayer(imgs[i - 1], imgs[i], pixels_ref, depth_ref, T_cur);
        T_cur_ref *= T_cur;
        puts("matrix corresponding to left img");
        cout << T_cur_ref.matrix() << endl;
        puts("matrix coresponding to last previous img");
        cout<<T_cur.matrix()<<endl;
        // DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    //
}

void DirectPoseEstimationSingleLayer( //my
    const cv::Mat& img1,
    const cv::Mat& img2,
    const VecVector2d& pixel_ref,
    const vector<double> depth_ref,
    Sophus::SE3d& T21)
{
    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0; // good projections
    VecVector2d goodProjection;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;
        goodProjection.clear();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero(); // 6x6 Hessian
        Vector6d b = Vector6d::Zero(); // 6x1 bias

        for (size_t i = 0; i < pixel_ref.size(); i++) {
            double u_ref = pixel_ref[i][0], v_ref = pixel_ref[i][1];

            double z_ref = depth_ref[i];
            double x_ref = (u_ref - cx) * z_ref / fx;
            double y_ref = (v_ref - cy) * z_ref / fy;

            auto T_21 = T21.matrix();
            Vector3d cur_position = T_21.block(0, 0, 3, 3) * (Vector3d) { x_ref, y_ref, z_ref } + T_21.block(0, 3, 1, 3);
            double cur_x = cur_position(0);
            double cur_y = cur_position(1);
            double cur_z = cur_position(2);
            float u = fx * cur_x / cur_z + cx;
            float v = fy * cur_y / cur_z + cy;
            if (u < half_patch_size || v < half_patch_size || //
                u > (img2.cols - half_patch_size) || v > (img2.rows - half_patch_size)) {
                continue;
            }
            nGood++;
            goodProjection.push_back(Vector2d(u, v));

            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, u_ref + x, v_ref + y) - GetPixelValue(img2, u + x, v + y);
                    Eigen::Vector2d J_img_pixel; // image gradients
                    J_img_pixel(0) = (GetPixelValue(img2, u + x + 1, v + y) - GetPixelValue(img2, u + x - 1, v + y)) / 2;
                    J_img_pixel(1) = (GetPixelValue(img2, u + x, v + y + 1) - GetPixelValue(img2, u + x, v + y - 1)) / 2;
                    Matrix26d partial_pixel_2_se3;
                    double z2 = pow(cur_z, 2);
                    partial_pixel_2_se3(0, 0) = fx / cur_z;
                    partial_pixel_2_se3(0, 1) = 0;
                    partial_pixel_2_se3(0, 2) = -fx * cur_x / z2;
                    partial_pixel_2_se3(0, 3) = -fx * cur_x * cur_y / z2;
                    partial_pixel_2_se3(0, 4) = fx + fx * cur_x * x / z2;
                    partial_pixel_2_se3(0, 5) = -fx * cur_y / cur_z;
                    partial_pixel_2_se3(1, 0) = 0;
                    partial_pixel_2_se3(1, 1) = fy / cur_z;
                    partial_pixel_2_se3(1, 2) = -fy * cur_y / z2;
                    partial_pixel_2_se3(1, 3) = -fy - fy * cur_y * y / z2;
                    partial_pixel_2_se3(1, 4) = fy * cur_x * cur_y / z2;
                    partial_pixel_2_se3(1, 5) = fy * cur_x / cur_z;
                    // total jacobian
                    Vector6d J = (J_img_pixel.transpose() * partial_pixel_2_se3).transpose();
                    H += J * J.transpose();
                    b += error * J;
                    cost += error * error;
                }
        }
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;
        cost /= nGood;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            // cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        //cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    // cout << "good projection: " << nGood << endl;
    // cout << "T21 = \n"
    //      << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    for (auto& px : pixel_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
            cv::Scalar(0, 250, 0));
    }
    for (auto& px : goodProjection) {
        cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
            cv::Scalar(0, 250, 0));
    }
    // cv::imshow("reference", img1_show);
    // cv::imshow("current", img2_show);
    // cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const VecVector2d& px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d& T21)
{
    // parameters
    int pyr_layer_num = 4;
    double pyramid_scale = 0.5;
    double scales[] = { 1.0, 0.5, 0.25, 0.125 };

    // create pyramids
    vector<Mat> pyr_img1, pyr_img2; // image pyramids
    pyr_img1.push_back(img1);
    pyr_img2.push_back(img2);
    for (int i = 1; i < pyr_layer_num; i++) {
        Mat img_pyr1, img_pyr2;
        resize(img1, img_pyr1, Size(pyr_img1[i - 1].cols * pyramid_scale, pyr_img1[i - 1].rows * pyramid_scale));
        resize(img2, img_pyr2, Size(pyr_img2[i - 1].cols * pyramid_scale, pyr_img2[i - 1].rows * pyramid_scale));
        pyr_img1.push_back(img_pyr1);
        pyr_img2.push_back(img_pyr2);
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy; // backup the old values
    for (int level = pyr_layer_num - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto& px : px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }
        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr_img1[level], pyr_img2[level], px_ref_pyr, depth_ref, T21);
    }
}
