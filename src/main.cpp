#include <PLS.hpp>
#include <Eigen/Dense>
#include <iostream>

using  std::cout;
using  std::endl;
using  Eigen::MatrixXd;

int main()
{
    Eigen::MatrixXd X(5, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0,
         10.0, 11.0, 12.0,
         13.0, 14.0, 15.0;

    Eigen::MatrixXd Y(5, 1);
    // generate a matrix that Y that have linear dependence on X
    for (int i = 0; i < 5; ++i) {
        Y(i, 0) = 2.0 * X(i, 0) + 3.0 * X(i, 1) - 1.0 * X(i, 2) 
		+ (static_cast<float>(rand()) / RAND_MAX - 0.5);
    }


    cout << "Original Y :\n" << Y << endl;

    PLS pls;

    pls.train(X, Y);
    Eigen::MatrixXd result = pls.predict(X);

    cout << "Predicted Results:\n" << result << endl;

    return 0;
}

