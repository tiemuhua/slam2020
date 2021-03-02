#include <eigen3/Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
    constexpr float fx = 277.34;
    constexpr float fy = 291.402;
    constexpr float cx = 312.234;
    constexpr float cy = 239.777;
    double camera_list[9] = { fx, 0, 0, 0, fy, 0, cx, cy, 1 };
    const Matrix3d camera = static_cast<const Matrix3d>(camera_list);
    cout<<camera<<endl;
    cout<<camera(1,1)<<endl;
}