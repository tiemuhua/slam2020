#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <string>

using namespace std;

int main(int argc, char **argv) {
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> exti_traj_poses, ground_traj_poses;

    /// implement pose reading code
    // start your code here (5~10 lines)
    double t, tx, ty, tz, qx, qy, qz, qw;
    ifstream esti_traj_fin("../estimated.txt");
    ifstream ground_truth_traj_fin("../groundtruth.txt");
    while (!esti_traj_fin.eof()) {
        static int cnt = 0;
        esti_traj_fin >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Quaterniond pose_q(qw, qx, qy, qz);
        pose_q.normalize();
        Eigen::Vector3d position(tx, ty, tz);
        exti_traj_poses.emplace_back(Sophus::SE3d(pose_q.toRotationMatrix(), position));
    }
    while (!ground_truth_traj_fin.eof()) {
        static int cnt = 0;
        ground_truth_traj_fin >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Quaterniond pose_q(qw, qx, qy, qz);
        pose_q.normalize();
        Eigen::Vector3d position(tx, ty, tz);
        ground_traj_poses.emplace_back(Sophus::SE3d(pose_q.toRotationMatrix(), position));
    }
    int size = ground_traj_poses.size();
    double error = 0;
    for (size_t i = 0; i < size; i++) {
        Sophus::SE3d ground_pose_SE3 = ground_traj_poses[i];
        Sophus::SE3d esti_pose_SE3 = exti_traj_poses[i];
        error += pow((ground_pose_SE3.inverse()*esti_pose_SE3).log().norm(), 2);
    }
    error = sqrt(error / size);
    cout << "error:\t" << error << endl;
    return 0;
}
