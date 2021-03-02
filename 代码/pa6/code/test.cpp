#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <iostream>
#include <string>
#include <vector>
int main(){
    using namespace Eigen;
    using namespace Sophus;
    Vector6d vec6d;
    vec6d<<1,1,1,0,0,0;
    SE3d se3;
    std::cout<<se3.matrix()<<std::endl;
}