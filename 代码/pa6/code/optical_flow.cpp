#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "../1.png"; // first image
string file_2 = "../2.png"; // second image

// TODO implement this funciton
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat& img1,
    const Mat& img2,
    const vector<KeyPoint>& kp1s,
    vector<KeyPoint>& kp2s,
    vector<bool>& success,
    bool inverse = false);

// TODO implement this funciton
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat& img1,
    const Mat& img2,
    const vector<KeyPoint>& kp1,
    vector<KeyPoint>& kp2,
    vector<bool>& success,
    bool inverse = false);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
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

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto& kp : kp1)
        pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel( //mine
    const Mat& img1,
    const Mat& img2,
    const vector<KeyPoint>& kp1s,
    vector<KeyPoint>& kp2s,
    vector<bool>& success,
    bool inverse)
{
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2s.empty();

    for (size_t i = 0; i < kp1s.size(); i++) {
        int x1 = kp1s[i].pt.x, y1 = kp1s[i].pt.y;
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2s[i].pt.x - x1;
            dy = kp2s[i].pt.y - y1;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            cout << "not inverse iter:" << iter << "\tlast cost \t" << lastCost << endl;
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (x1 + dx - half_patch_size < 0 || x1 + dx + half_patch_size + 1 >= img1.cols || //
                y1 + dy - half_patch_size < 0 || y1 + dy + half_patch_size + 1 >= img1.rows || //
                x1 - 1 - half_patch_size < 0 || x1 + 1 + half_patch_size + 1 >= img1.cols || //
                y1 - 1 - half_patch_size < 0 || y1 + 1 + half_patch_size + 1 >= img1.rows) { // go outside
                succ = false;
                break;
            }
            // compute cost and jacobian
            for (int x = -half_patch_size; x <= half_patch_size; ++x) {
                for (int y = -half_patch_size; y <= half_patch_size; ++y) {
                    double error = GetPixelValue(img1, x1 + x, y1 + y) - GetPixelValue(img2, x1 + dx + x, y1 + dy + y);
                    Eigen::Vector2d J;
                    /**
                     * just as the file "test_addressing_time.cpp" shows
                     * addressing is a very time-costing operation
                     * it is not very worth-well for us to use a "if" control statement,
                     * in order to decrease a small matrix multiple
                    */
                    if (inverse) {
                        J[0] = (GetPixelValue(img1, x1 + x + 1, y1 + y) - GetPixelValue(img1, x1 + x - 1, y1 + y)) / 2;
                        J[1] = (GetPixelValue(img1, x1 + x, y1 + y + 1) - GetPixelValue(img1, x1 + x, y1 + y - 1)) / 2;
                    } else {
                        J[0] = (GetPixelValue(img2, x1 + dx + x + 1, y1 + dy + y) - GetPixelValue(img2, x1 + dx + x - 1, y1 + dy + y)) / 2;
                        J[1] = (GetPixelValue(img2, x1 + dx + x, y1 + dy + y + 1) - GetPixelValue(img2, x1 + dx + x, y1 + dy + y - 1)) / 2;
                    }
                    H += J * J.transpose();
                    b += error * J;
                    cost += error * error;
                }
            }
            Eigen::Vector2d update = H.ldlt().solve(b);
            dx += update[0];
            dy += update[1];

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }
            lastCost = cost;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2s[i].pt = kp1s[i].pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp1s[i];
            tracked.pt += cv::Point2f(dx, dy);
            kp2s.push_back(tracked);
        }
    }
}
/* my*/
void OpticalFlowMultiLevel(
    const Mat& img1,
    const Mat& img2,
    const vector<KeyPoint>& kp1s,
    vector<KeyPoint>& kp2s,
    vector<bool>& success,
    bool inverse)
{
    // parameters
    int pyr_layer_num = 4;
    double pyramid_scale = 0.5;

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

    vector<KeyPoint> pyr_pk1s, pyr_pk2s;
    for (int i = 0; i < kp1s.size(); ++i) {
        KeyPoint pyr_kp = kp1s[i];
        pyr_kp.pt *= pow(pyramid_scale, pyr_layer_num - 1);
        pyr_pk1s.push_back(pyr_kp);
        pyr_pk2s.push_back(pyr_kp);
    }

    for (int i = pyr_layer_num - 1; i >= 0; --i) {
        success.clear();
        OpticalFlowSingleLevel(pyr_img1[i], pyr_img2[i], pyr_pk1s, pyr_pk2s, success, true);
        if (i != 0) {
            for (int i = 0; i < pyr_pk1s.size(); i++) {
                pyr_pk1s[i].pt /= pyramid_scale;
                pyr_pk2s[i].pt /= pyramid_scale;
            }
        }
    }

    for (auto kp : pyr_pk2s) {
        kp2s.push_back(kp);
    }
}
