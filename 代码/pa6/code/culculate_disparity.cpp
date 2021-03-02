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

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double baseline = 0.573;

string left_str = "../left.png";
string right_str = "../right.png";
string disparity_str = "../disparity.png";

inline float GetPixelValue(const cv::Mat& img, float x, float y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
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
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>>& pointcloud);

int main()
{
    Mat left_img = imread(left_str, 0);
    Mat right_img = imread(right_str, 0);
    Mat disparity_img = imread(disparity_str, 0);
    int cols = left_img.cols, rows = left_img.rows;
    int col_segment = 20, row_segment = 6;
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(left_img, kp1);

    vector<KeyPoint> kp2;
    vector<bool> success;
    OpticalFlowMultiLevel(left_img, right_img, kp1, kp2, success, true);

    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> true_points, culc_points;

    for (int i = 0; i < kp1.size(); ++i) {
        float true_disparity = GetPixelValue(disparity_img, kp1[i].pt.x, kp1[i].pt.y);
        float culc_disparity = (kp1[i].pt - kp2[i].pt).x, y = (kp1[i].pt - kp2[i].pt).y;
        if (culc_disparity < 5 || y > 2 || y < -2) {
            continue;
        }
        cout << "true_disparity\t" << true_disparity << "\tculc_disparity\t" << culc_disparity << endl;
        Vector4d true_point, culc_point;
        double true_z = baseline * fx / true_disparity;
        true_point(2)=true_z;
        true_point(1)=(kp1[i].pt.y-cy)/fy*true_z;
        true_point(0)=(kp1[i].pt.x-cx)/fx*true_z;

        double culc_z = baseline * fx / culc_disparity;
        culc_point(2)=culc_z;
        culc_point(1)=(kp1[i].pt.y-cy)/fy*culc_z;
        culc_point(0)=(kp1[i].pt.x-cx)/fx*culc_z;
        true_points.push_back(true_point);
        culc_points.push_back(culc_point);
    }
    showPointCloud(true_points);
    // showPointCloud(culc_points);
}


void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>>& pointcloud)
{
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto& p : pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000); // sleep 5 ms
    }
    return;
}
