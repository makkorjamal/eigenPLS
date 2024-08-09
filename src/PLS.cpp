#include <PLS.hpp>
#include <iostream>

using  std::endl;
using  std::cerr;
using Eigen::MatrixXd;
using Eigen::VectorXd;

PLS::PLS( )
{
	// nothing to do
}


PLS::PLS(
	const MatrixXd &B,
	const MatrixXd &meanX,
	const MatrixXd &meanY ) : B(B), mean0X(meanX), mean0Y(meanY)
{
	// nothing to do
}

PLS::~PLS()
{
	// nothing to do
}

void PLS::train(
	const MatrixXd &Xdata,
	const MatrixXd &Ydata,
	double epsilon )
{
	if (Xdata.rows() != Ydata.rows()){

		cerr<<"X size is different from Y "<<Xdata.size()<<" # "<<Ydata.size()<<endl;
		return;
	}

	MatrixXd X, Y;
	X = Xdata;
	mean0X = X.colwise().mean();
	Y = Ydata;
	mean0Y = Y.colwise().mean();

	for (int i = 0; i < X.rows(); ++i) 
		X.row(i) = X.row(i) - mean0X;

	for (int i = 0; i < Y.rows(); ++i) 
		Y.row(i) = Y.row(i) - mean0Y;

        MatrixXd T, U, W, C, P, Q, Bdiag;
        VectorXd t, w, u_old,e, c, p, q, b;
	VectorXd u = VectorXd::Random(X.rows());
	while (1)
	{
		while (1)
		{
			//maximizing infomation content in from X and Y
			w = X.transpose() * u;
			w = w / w.norm();
			t = X * w; // latent vector in X
			t = t / t.norm();
			c = Y.transpose() * t;
			c = c / c.norm();
			u_old = u;
			u = Y * c; // latent vector in Y
			e = u- u_old; // try to minimise the error
			double error = e.norm();
			if (error < epsilon) break;
		}
		b = t.transpose() * u;
		assert(b.cols() == 1 && b.rows() == 1);
		if (C.cols() == 0)
			C = c;
		else
			C.conservativeResize(C.rows(), C.cols() + 1); 
			C.col(C.cols() - 1) = c; 

		double temp = t.norm();
		p = X.transpose() * t / (temp * temp);
		if (P.cols() == 0)
			P = p;
		else
			P.conservativeResize(P.rows(), P.cols() + 1); 
			P.col(P.cols() - 1) = p; 
		if (Bdiag.cols() == 0)
			Bdiag = b;
		else
			Bdiag.conservativeResize(Bdiag.rows(), Bdiag.cols() + 1); 
			Bdiag.col(Bdiag.cols() - 1) = b; 

		X = X - t * p.transpose();
		Y = Y - t * c.transpose();
		if (X.norm() < 0.001) break;
	}
	Bdiag = Bdiag.diagonal();
	B = pseudoMat(P);
	B = B * Bdiag;
	B = B * C.transpose();
}
MatrixXd PLS::pseudoMat(MatrixXd P)
{
	MatrixXd P_transpose = P.transpose();
	MatrixXd pseudoM = P_transpose.completeOrthogonalDecomposition().pseudoInverse();
	return pseudoM;
}


const MatrixXd &PLS::getB() const
{
	return this->B;
}


const MatrixXd &PLS::getMeanX() const
{
	return this->mean0X;
}


const MatrixXd &PLS::getMeanY() const
{
	return this->mean0Y;
}


MatrixXd PLS::predict(
	const MatrixXd &v ) const
{
	MatrixXd temp;
	MatrixXd result = MatrixXd::Zero(v.rows(),B.cols());	
	temp = v;

	for (int i = 0; i < temp.rows(); ++i)
	{
		// subtract the training X mean
		temp.row(i) -= mean0X;
		// predict the Y matrix
		result.row(i) = temp.row(i) * B;
		// add the training X mean
		result.row(i) += mean0Y;
	}

	return result;
}
