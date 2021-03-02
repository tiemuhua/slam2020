//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

using namespace Eigen;

#include <sophus/so3.hpp>

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
        0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
        -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    auto E_svd=E.jacobiSvd(ComputeThinU | ComputeThinV);
    Matrix3d V = E_svd.matrixV(), U = E_svd.matrixU();
    Vector3d S = E_svd.singularValues();
    float eigen_value1 = S(0), eigen_value2 = S(1);
    S(0) = S(1) = (eigen_value1 + eigen_value2) / 2;
    Matrix3d Sigma=S.asDiagonal();

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;
    Matrix3d RZ=AngleAxisd(M_PI_2,Vector3d(0,0,1)).matrix();
    // SVD and fix sigular values
    // START YOUR CODE HERE
    // END YOUR CODE HERE

    // set t1, t2, R1, R2
    // START YOUR CODE HERE
    Matrix3d t_wedge1=U*RZ*Sigma*U.transpose();
    Matrix3d t_wedge2=U*RZ.transpose()*Sigma*U.transpose();

    Matrix3d R1=U*RZ.transpose()*V.transpose();
    Matrix3d R2=U*RZ*V.transpose();
    // END YOUR CODE HERE

    cout << "R1 = \n" << R1 << endl;
    cout << "R2 = \n" << R2 << endl;
    cout << "t1 = \n" << Sophus::SO3d::vee(t_wedge1) << endl;
    cout << "t2 = \n" << Sophus::SO3d::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge2 * R2;
    cout << "t^R = \n" << tR << endl;

    return 0;
}