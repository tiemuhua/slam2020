#include <bits/stdc++.h>
using namespace std;
char buffer[250];
int main1() {
    double dist;
    double x[100], y[100];
    int tot = 1;
    ifstream infile("trajectory.txt");
    ofstream outfile("result.txt");
    while (!infile.eof()) {
        infile.getline(buffer, 20);  //整行读入
        sscanf(buffer, "%lf %lf", &x[tot], &y[tot]);
        tot++;
    }
    tot--;
    for (int i = 1; i < tot; i++) {
        dist = sqrt(pow(x[i] - x[i + 1], 2) + pow(y[i] - y[i + 1], 2));
        outfile << dist << endl;
    }
    infile.close();
    outfile.close();
    return 0;
}
int main(){
    freopen("trajectory.txt","r",stdin);
    for(int i=0;i<100;++i){
        double tmp;
        cin>>tmp;
        cout<<tmp<<endl;
    }
}