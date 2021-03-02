//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.hpp"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "../p3d.txt";
string p2d_file = "../p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    ifstream p2d_in(p2d_file),p3d_in(p3d_file);
    double tmp1,tmp2,tmp3;
    while (!p2d_in.eof()) {
        p2d_in>>tmp1>>tmp2;
        p2d.emplace_back(Vector2d(tmp1,tmp2));
    }
    while(!p3d_in.eof()){
        p3d_in>>tmp1>>tmp2>>tmp3;
        p3d.emplace_back(Vector3d(tmp1,tmp2,tmp3));
    }
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    std::cout << "points: " << nPoints << endl;

    Sophus::SE3d T_esti; // estimated pose
    Vector6d pose_se3=Vector6d::Zero();
    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            Matrix4d camera_pose_4d=Sophus::SE3d::exp(pose_se3).matrix();
            Vector3d p_camera = camera_pose_4d.block(0,0,3,3) * p3d[i]+camera_pose_4d.block(0,3,3,1);
            double s = p_camera(2);
            Vector2d e = p2d[i] - (K * p_camera).block(0,0,2,1) / s;
            cost+=e.norm();

	        // compute jacobian
            Matrix<double, 2, 6> J;
            double x=p_camera(0),y=p_camera(1),z=p_camera(2);
            double z2=pow(z,2);
            J(0,0)=fx/z;
            J(0,1)=0;
            J(0,2)=-fx*x/z2;
            J(0,3)=-fx*x*y/z2;
            J(0,4)=fx+fx*x*x/z2;
            J(0,5)=-fx*y/z;
            J(1,0)=0;
            J(1,1)=fy/z;
            J(1,2)=-fy*y/z2;
            J(1,3)=-fy-fy*y*y/z2;
            J(1,4)=fy*x*y/z2;
            J(1,5)=fy*x/z;
            J*=-1;
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

	    // solve dx 
        pose_se3 += H.colPivHouseholderQr().inverse()*b;
        

        if (isnan(pose_se3[0])) {
            cout << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        lastCost = cost;
        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << Sophus::SE3d::exp(pose_se3).matrix() << endl;
    return 0;
}
