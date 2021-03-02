#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include<iostream>
int main()
{
    Eigen::Matrix<double, 6, 4> mat;
    mat
        << -1,
        -1, 0, 0,
        0, -1, 1, 0,
        0, 0, -1, 1,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    Eigen::Matrix4d matTmat=mat.transpose()*mat;
    std::cout<<matTmat.inverse()*mat.transpose()<<std::endl;
}