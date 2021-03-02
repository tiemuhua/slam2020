#include <bits/stdc++.h>

#include <Eigen/Cholesky>
#include <Eigen/QR>

using namespace std;

int main() {
    const int matDimention = 100;
    srand((unsigned)time(NULL));
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(matDimention, matDimention);
    Eigen::VectorXd b = Eigen::VectorXd::Random(matDimention);
    clock_t t1 = clock();
    Eigen::VectorXd ans = A.colPivHouseholderQr().solve(b);
    clock_t t2 = clock();
    Eigen::MatrixXd ATA = A * A.transpose();
    ans = ATA.llt().solve(A.transpose() * b);
    clock_t t3 = clock();
    cout << "householder qr time\t" << t2 - t1 << endl;
    cout << "llt time\t" << t3 - t2 << endl;
}