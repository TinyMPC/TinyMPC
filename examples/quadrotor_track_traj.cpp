#include <iostream>
#include <Eigen/Dense>
#include <tinympc/tinympc.hpp>

using Eigen::Matrix;

int main() {
    Matrix<tinytype,2,2> m;
    Matrix<tinytype,2,4> b;
    Matrix<tinytype,2,4> c;
    Matrix<tinytype,4,2> mt;
    Matrix<tinytype,2,2> d;
    m(0,0) = 1;
    m(0,1) = 2;
    m(1,0) = 3;
    m(1,1) = 4;
    b << 1, 2, 3, 4, 5, 6, 7, 8;
    std::cout << m << std::endl;
    std::cout << b << std::endl;
    c = m*b;
    mt = b.transpose();
    std::cout << c << std::endl;
    std::cout << mt << std::endl;

    std::cout << d << std::endl;

    return 0;
}