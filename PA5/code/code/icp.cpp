#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <string>

using namespace std;
using namespace Eigen;
typedef Matrix<double, 6, 1> Vector6d;
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses);
int main(){
    ifstream fin("../compare.txt");
    vector<double> time1s,time2s;
    vector<Vector3d> p1,p2,so31,so32;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses;
    while(!fin.eof()){
        double time1,time2,p10,p11,p12,p20,p21,p22,q10,q11,q12,q13,q20,q21,q22,q23;
        fin>>time1>>p10>>p11>>p12>>q11>>q12>>q13>>q10>>time2>>p20>>p21>>p22>>q21>>q22>>q23>>q20;
        time1s.push_back(time1);
        time2s.push_back(time2);
        p1.emplace_back(Vector3d(p10,p11,p12));
        p2.emplace_back(Vector3d(p20,p21,p22));
    }
    int size=time1s.size();
    vector<Vector3d> new_q2,new_p2;
    for(int i=1;i<size-1;++i){
        double lambda=(time2s[i+1]-time1s[i])/(time2s[i+1]-time2s[i-1]);
        new_p2.emplace_back(p2[i-1]*lambda+p2[i+1]*(1-lambda));
    }
    p2=new_p2;

    Vector3d p10=Vector3d::Zero(),p20=Vector3d::Zero();
    for(int i=0;i<size;++i){
        p10+=p1[i];
        p20+=p2[i];
    }
    p10/=size;
    p20/=size;

    Matrix3d W=Matrix3d::Zero();
    for(int i=0;i<size;++i){
        W+=(p2[i]-p20)*(p1[i]-p10).transpose();
    }
    auto W_svd=W.jacobiSvd(ComputeThinU | ComputeThinV);
    Matrix3d U=W_svd.matrixU(),V=W_svd.matrixV();
    Matrix3d R=V*U.transpose();
    Vector3d t=p10-R*p20;
    
    for(int i=0;i<size;++i){
        p2[i]=R*p2[i]+t;
    }

    for(int i=0;i<size;++i){
        poses.emplace_back(Matrix3d::Identity(),p1[i]);
    }
    for(int i=0;i<size;++i){
        //cout<<p2[i].transpose()<<endl;
        poses.emplace_back(Matrix3d::Identity(),p2[i]);
    }
    DrawTrajectory(poses);
}

void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses) {
    if (poses.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float)i / poses.size(), 0.0f, (float)i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);  // sleep 5 ms
    }
}