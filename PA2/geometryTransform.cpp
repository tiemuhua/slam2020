#include <bits/stdc++.h>
#include <Eigen/Geometry>
using namespace std;
using namespace Eigen;
int main1() {
    Quaterniond q1 = {0.55, 0.3, 0.2, 0.2};
    q1.normalize();
    Quaterniond q2 = {-0.1, 0.3, -0.7, 0.2};
    q2.normalize();

    Matrix3d R1 = q1.toRotationMatrix();
    Matrix3d R2 = q2.toRotationMatrix();
    Vector3d p1 = {0.7, 1.1, 0.2};
    Vector3d p2 = {-0.1, 0.4, 0.8};

    Matrix4d world2robot1;
    world2robot1.block(0,3,3,1)=p1;
    world2robot1.block(0,0,3,3)=R1;
    world2robot1(3,3)=1;
    Matrix4d robot12world=world2robot1.inverse();

    Matrix4d world2robot2;
    world2robot2.block(0,3,3,1)=p2;
    world2robot2.block(0,0,3,3)=R1;
    world2robot2(3,3)=1;
    Matrix4d robot22world=world2robot2.inverse();
    
    Vector4d p_in1 = {0.5, -0.1, 0.2,1};
    Vector4d p_in_world = robot12world*p_in1;
    Vector4d p_in2 = world2robot2 * p_in_world;
    cout<<p_in2<<endl;
}
int main() {
    Quaterniond q1 = {0.55, 0.3, 0.2, 0.2};
    q1.normalize();
    Quaterniond q2 = {-0.1, 0.3, -0.7, 0.2};
    q2.normalize();

    Matrix3d R1 = q1.toRotationMatrix();
    Matrix3d R2 = q2.toRotationMatrix();
    Vector3d p1 = {0.7, 1.1, 0.2};
    Vector3d p2 = {-0.1, 0.4, 0.8};

    Matrix4d robot1_2_world;
    robot1_2_world.block(0,3,3,1)=p1;
    robot1_2_world.block(0,0,3,3)=R1;
    robot1_2_world(3,3)=1;
    Matrix4d world_2_robot1=robot1_2_world.inverse();

    Matrix4d robot2_2_world;
    robot2_2_world.block(0,3,3,1)=p2;
    robot2_2_world.block(0,0,3,3)=R1;
    robot2_2_world(3,3)=1;
    Matrix4d world_2_robot2=robot2_2_world.inverse();
    
    Vector4d p_in1 = {0.5, -0.1, 0.2,1};
    Vector4d p_in_world = robot1_2_world*p_in1;
    Vector4d p_in2 = world_2_robot2 * p_in_world;
    cout<<p_in2<<endl;
}