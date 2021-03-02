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

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "../poses.txt";
string points_file = "../points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

Matrix3d camera;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat& img, float x, float y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}

// g2o vertex that use sophus::SE3 as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSophus() {}
    ~VertexSophus() {}
    bool read(std::istream& is) {}
    bool write(std::ostream& os) const {}

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

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<float, 16, 1> Vector16f;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16f, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(cv::Mat& target)
    {
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override
    {
        Vector3d point = static_cast<const VertexSBAPointXYZ*>(_vertices[0])->estimate();
        Sophus::SE3d SE3 = static_cast<const VertexSophus*>(_vertices[1])->estimate();
        Vector3d point_in_camera = SE3.rotationMatrix() * point + SE3.translation();
        Vector3d projected_point = camera * point_in_camera;
        double u0 = projected_point(0) / projected_point(2), v0 = projected_point(1) / projected_point(2);
        Vector16f cur_color;
        if (u0 < 2 || v0 < 2 || u0 + 1 >= targetImg.cols || v0 + 1 >= targetImg.rows) {
            return;
        }
        for (int u = -2; u <= 1; u++) {
            for (int v = -2; v <= 1; v++) {
                cur_color((u + 2) * 4 + v + 2) = GetPixelValue(targetImg, u + u0, v + v0);
            }
        }
        _error = measurement() - cur_color;
    }

    // Let g2o compute jacobian for you
    virtual bool read(istream& in) {}
    virtual bool write(ostream& out) const {}

private:
    cv::Mat targetImg; // the target image
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3& poses, const VecVec3d& points);

int main(int argc, char** argv)
{
    camera << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0)
            break;
        double data[7];
        for (auto& d : data)
            fin >> d;
        auto q=Eigen::Quaterniond(data[6], data[3], data[4], data[5]);
        q.normalize();
        poses.push_back(Sophus::SE3d(
            q,
            Eigen::Vector3d(data[0], data[1], data[2])));
        if (!fin.good())
            break;
    }
    fin.close();

    vector<float*> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = { 0 };
        for (int i = 0; i < 3; i++)
            fin >> xyz[i];
        if (xyz[0] == 0)
            break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float* c = new float[16];
        for (int i = 0; i < 16; i++)
            fin >> c[i];
        color.push_back(c);

        if (fin.good() == false)
            break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("./%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock; // 求解的向量是6＊1的
    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolver;
    unique_ptr<LinearSolver> linear_solver_ptr;
    unique_ptr<DirectBlock> direct_block_solver_ptr(new DirectBlock(move(linear_solver_ptr)));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(move(direct_block_solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    const size_t pose_size = images.size();
    const size_t point_size = points.size();

    vector<VertexSBAPointXYZ*> vec_vertex_point;
    vector<VertexSophus*> vec_vertex_se3;

    for (size_t point_i; point_i < point_size; ++point_i) {
        VertexSBAPointXYZ* v_point = new VertexSBAPointXYZ();
        v_point->setEstimate(points[point_i]);
        v_point->setId(point_i);
        vec_vertex_point.push_back(v_point);
        optimizer.addVertex(v_point);
    }

    for (size_t pose_i = 0; pose_i < pose_size; ++pose_i) {
        VertexSophus* v_se3 = new VertexSophus();
        v_se3->setEstimate(poses[pose_i]);
        v_se3->setId(point_size + pose_i);
        vec_vertex_se3.push_back(v_se3);
        optimizer.addVertex(v_se3);
    }
    
    // TODO add vertices, edges into the graph optimizer
    for (size_t point_i = 0; point_i < point_size; ++point_i) {
        auto color_ptr = color[point_i];
        for (size_t pose_i=0;pose_i<pose_size;++pose_i) {
            EdgeDirectProjection* e = new EdgeDirectProjection(images[pose_i]);
            e->vertices()[0] = static_cast<OptimizableGraph::Vertex*>(vec_vertex_point[point_i]);
            e->vertices()[1] = static_cast<OptimizableGraph::Vertex*>(vec_vertex_se3[pose_i]);
            e->setMeasurement(static_cast<Vector16f>(color_ptr));
            optimizer.addEdge(e);
        }
    }

    // perform optimization
    optimizer.initializeOptimization(0);
    //optimizer.optimize(200);

    // TODO fetch data from the optimizer
    points.clear();
    poses.clear();
    for(size_t point_i=0;point_i<point_size;++point_i){
        points.push_back(static_cast<VertexSBAPointXYZ*>(optimizer.vertices().find(point_i)->second)->estimate());
    }
    for(size_t pose_i=0;pose_i<pose_size;++pose_i){
        poses.push_back(static_cast<VertexSophus*>(optimizer.vertices().find(pose_i+pose_size)->second)->estimate());
    }

    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
    for (auto& c : color)
        delete[] c;
    return 0;
}

void Draw(const VecSE3& poses, const VecVec3d& points)
{
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
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
