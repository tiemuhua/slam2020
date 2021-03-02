#include <iostream>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace g2o;
using namespace Sophus;
using namespace Eigen;
using namespace std;
using namespace cv;

typedef std::vector<SE3d, aligned_allocator<SE3d>> VecSE3;
typedef std::vector<Vector3d, aligned_allocator<Vector3d>> VecVec3d;
typedef Eigen::Matrix<double, 16, 1> Vector16d;

// global variables
std::string pose_file = "../poses.txt";
std::string points_file = "../points.txt";

// intrinsics
constexpr float fx = 277.34;
constexpr float fy = 291.402;
constexpr float cx = 312.234;
constexpr float cy = 239.777;
double camera_list[9] = { fx, 0, 0, 0, fy, 0, cx, cy, 1 };
const Matrix3d camera = static_cast<const Matrix3d>(camera_list);

// bilinear interpolation
inline float GetPixelValue(const Mat& img, float x, float y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}

class VertexSophus : public BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSophus() {}
    ~VertexSophus() {}
    bool read(std::istream& in) {}
    bool write(std::ostream& out) const {}
    virtual void setToOriginImpl()
    {
        _estimate = Sophus::SE3d();
    }
    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeDirectProjection(cv::Mat& img)
        : _img(img)
    {
    }
    ~EdgeDirectProjection() {}
    virtual void computeError() override
    {
        static long long cnt = 0;
        Vector3d point = static_cast<const VertexSBAPointXYZ*>(_vertices[0])->estimate();
        Sophus::SE3d pose = static_cast<const VertexSophus*>(_vertices[1])->estimate();
        Vector3d pose_in_camera_coor = pose.rotationMatrix() * point + pose.translation();
        Vector3d pose_project_in_camera = camera * pose_in_camera_coor;
        float u0 = pose_project_in_camera(0) / pose_project_in_camera(2);
        float v0 = pose_project_in_camera(1) / pose_project_in_camera(2);
        int index = 0;
        for (int u = -2; u <= 1; u++) {
            for (int v = -2; v <= 1; v++) {
                if (u0 + u < 0 || v0 + v < 0 || u0 + u > _img.cols || v0 + v > _img.rows) {
                    _error[index] = 0;
                    this->setLevel(1);
                    continue;
                }
                float cur_color = GetPixelValue(_img, u0 + u, v0 + v);
                _error[index] = _measurement[index] - cur_color;
                index++;
            }
        }
        if(_error.norm()>1){
            cnt++;
            cout<<"cnt\t"<<cnt<<"\terror norm\t"<<_error.norm()<<endl;
        }
    }

    virtual void linearizeOplus()
    {
        Vector3d point = static_cast<const VertexSBAPointXYZ*>(_vertices[0])->estimate();
        Sophus::SE3d pose = static_cast<const VertexSophus*>(_vertices[1])->estimate();
        Vector3d pose_in_camera_coor = pose * point;
        Vector3d pose_project_in_camera = camera * pose_in_camera_coor;

        float u0 = pose_project_in_camera(0) / pose_project_in_camera(2);
        float v0 = pose_project_in_camera(1) / pose_project_in_camera(2);
        double x = pose_in_camera_coor(0), y = pose_in_camera_coor(1), z = pose_in_camera_coor(2);
        double inv_z = 1.0 / z, inv_z2 = 1.0 / z / z;

        //u，v分别对x,y,z求导
        Eigen::Matrix<double, 2, 3> partial_uv_2_xyz;
        partial_uv_2_xyz(0, 0) = fx * inv_z;
        partial_uv_2_xyz(0, 1) = 0;
        partial_uv_2_xyz(0, 2) = -fx * x * inv_z2;
        partial_uv_2_xyz(1, 0) = 0;
        partial_uv_2_xyz(1, 1) = fy * inv_z;
        partial_uv_2_xyz(1, 2) = -fy * y * inv_z2;

        Eigen::Matrix<double, 3, 6> partial_poseInCamera_2_se3 = Eigen::Matrix<double, 3, 6>::Zero();
        partial_poseInCamera_2_se3(0, 0) = 1;
        partial_poseInCamera_2_se3(0, 4) = z;
        partial_poseInCamera_2_se3(0, 5) = -y;
        partial_poseInCamera_2_se3(1, 1) = 1;
        partial_poseInCamera_2_se3(1, 3) = -z;
        partial_poseInCamera_2_se3(1, 5) = x;
        partial_poseInCamera_2_se3(2, 2) = 1;
        partial_poseInCamera_2_se3(2, 3) = y;
        partial_poseInCamera_2_se3(2, 4) = -x;

        Eigen::Matrix<double, 1, 2> partial_I_2_uv;
        for (int i = -2; i < 2; i++)
            for (int j = -2; j < 2; j++) {
                int num = 4 * i + j + 10;
                if (u0 + i < 0 || u0 + i >= _img.cols || v0 + j < 0 || v0 + j >= _img.rows) {
                    continue;
                }
                partial_I_2_uv(0, 0) = (GetPixelValue(_img, u0 + i + 1, v0 + j) - GetPixelValue(_img, u0 + i - 1, v0 + j)) / 2;
                partial_I_2_uv(0, 1) = (GetPixelValue(_img, u0 + i, v0 + j + 1) - GetPixelValue(_img, u0 + i, v0 + j - 1)) / 2; //
                _jacobianOplusXi.block<1, 3>(num, 0) = -partial_I_2_uv * partial_uv_2_xyz * pose.rotationMatrix();
                _jacobianOplusXj.block<1, 6>(num, 0) = -partial_I_2_uv * partial_uv_2_xyz * partial_poseInCamera_2_se3;
            }
    }
    virtual bool read(istream& in) {}
    virtual bool write(ostream& out) const {}

private:
    cv::Mat _img;
};
void Draw(const char* Title, const VecSE3& poses, const VecVec3d& points);

int main()
{
    std::ios::sync_with_stdio(false);
    VecSE3 poses, poses0;
    VecVec3d points, points0;
    ifstream fin(pose_file);
    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) {
            break;
        }
        double se3_data[7];
        for (auto& d : se3_data) {
            fin >> d;
        }
        auto q = Eigen::Quaterniond(se3_data[6], se3_data[3], se3_data[4], se3_data[5]);
        q.normalize();
        poses.push_back(Sophus::SE3d(q, Vector3d(se3_data[0], se3_data[1], se3_data[2])));
        if (!fin.good())
            break;
    }
    fin.close();

    vector<Vector16d> colors;
    fin.open(points_file);
    int cnt=0;
    while (!fin.eof()) {
        double x, y, z;
        fin >> x >> y >> z;
        cnt++;
        cout<<x<<"\t"<<y<<"\t"<<z<<"\t"<<cnt<<endl;
        if (x == 0)
            break;
        points.push_back(Vector3d(x, y, z));
        Vector16d color;
        for (int i = 0; i < 16; ++i) {
            fin >> color[i];
        }
        colors.push_back(color);
        if (!fin.good()) {
            break;
        }
    }
    fin.close();
    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    poses0.assign(poses.begin(), poses.end());
    points0.assign(points.begin(), points.end());

    vector<cv::Mat> images;
    boost::format fmt("../%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock; // 求解的向量是6＊1的
    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolver;
    unique_ptr<LinearSolver> linear_solver_ptr(new LinearSolver());
    unique_ptr<DirectBlock> direct_block_solver_ptr(new DirectBlock(move(linear_solver_ptr)));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(move(direct_block_solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    const size_t points_size = points.size();
    const size_t poses_size = poses.size();
    vector<VertexSBAPointXYZ*> point_vectex_ptres;
    vector<VertexSophus*> pose_vertex_ptres;

    for (int point_id = 0; point_id < points_size; point_id++) {
        VertexSBAPointXYZ* point_vertex = new VertexSBAPointXYZ();
        point_vertex->setEstimate(points[point_id]);
        point_vertex->setId(point_id);
        point_vertex->setMarginalized(true);
        point_vectex_ptres.push_back(point_vertex);
        optimizer.addVertex(point_vertex);
    }

    for (int pose_id = 0; pose_id < poses_size; pose_id++) {
        VertexSophus* pose_vertex = new VertexSophus();
        pose_vertex->setEstimate(poses[pose_id]);
        pose_vertex->setId(points_size + pose_id);
        pose_vertex_ptres.push_back(pose_vertex);
        if (pose_id == 0)
            pose_vertex->setFixed(true);
        optimizer.addVertex(pose_vertex);
    }

    int index = 0;
    for (int point_id = 0; point_id < points_size; point_id++) {
        for (int pose_id = 0; pose_id < poses_size; pose_id++) {
            EdgeDirectProjection* e = new EdgeDirectProjection(images[pose_id]);
            e->vertices()[0] = static_cast<OptimizableGraph::Vertex*>(point_vectex_ptres[point_id]);
            e->vertices()[1] = static_cast<OptimizableGraph::Vertex*>(pose_vertex_ptres[pose_id]);
            e->setMeasurement(colors[point_id]);
            e->setInformation(Eigen::Matrix<double, 16, 16>::Identity());
            RobustKernelHuber* rk = new RobustKernelHuber();
            rk->setDelta(1.0);
            e->setRobustKernel(rk);
            e->setId(index);
            optimizer.addEdge(e);
            index++;
        }
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);

    poses.clear();
    points.clear();
    for (size_t i = 0; i < points_size; i++) {
        g2o::VertexSBAPointXYZ* point_XYZ_new = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i));
        Eigen::Vector3d point = point_XYZ_new->estimate();
        points.push_back(point);
    }
    for (size_t i = 0; i < poses_size; i++) {
        VertexSophus* pose_vertex_new = dynamic_cast<VertexSophus*>(optimizer.vertex(i
            + points_size));
        Sophus::SE3d pose = pose_vertex_new->estimate();
        poses.push_back(pose);
    }
    // plot the optimized points and poses
    Draw("Data Before Optimization", poses0, points0);
    Draw("Optimized Data", poses, points);
}

void Draw(const char* Title, const VecSE3& poses, const VecVec3d& points)
{
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(Title, 1024, 768);
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
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto& Tcw : poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat*)m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2] / 4, 1.0 - points[i][2] / 4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000); // sleep 5 ms
    }
}