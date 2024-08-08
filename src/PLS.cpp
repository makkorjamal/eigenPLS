#include <PLS.hpp>
#include <iostream>
#include <random>
#include <math.h>
#include <cstdio>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;

PLS::PLS( )
{
	// nothing to do
}


PLS::PLS(
	const MatrixXf &B,
	const MatrixXf &meanX,
	const MatrixXf &meanY ) : B(B), mean0X(meanX), mean0Y(meanY)
{
	// nothing to do
}

PLS::~PLS()
{
	// nothing to do
}

void PLS::train(
	const MatrixXf &Xdata,
	const MatrixXf &Ydata,
	float epsilon )
{
	if (Xdata.rows() != Ydata.rows()){

		cerr<<"X size is different from Y "<<Xdata.size()<<" # "<<Ydata.size()<<endl;
		return;
	}

	MatrixXf X, Y;
	X = Xdata;
	mean0X = X.colwise().mean();
	Y = Ydata;
	mean0Y = Y.colwise().mean();

	for (int i = 0; i < X.rows(); ++i) 
		X.row(i) = X.row(i) - mean0X;

	for (int i = 0; i < Y.rows(); ++i) 
		Y.row(i) = Y.row(i) - mean0Y;

        MatrixXf T, U, W, C, P, Q, Bdiag;
        VectorXf t, w, u_old,e, c, p, q, b;
	VectorXf u = VectorXf::Random(X.rows());
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
			float error = e.norm();
			if (error < epsilon) break;
		}
		b = t.transpose() * u;
		assert(b.cols() == 1 && b.rows() == 1);
#if (0)
		if (T.cols() == 0)
			T = t;
		else
			T.conservativeResize(T.rows(), T.cols() + 1); 
			T.col(T.cols() - 1) = t; 

		if (U.cols() == 0)
			U = u;
		else
			U.conservativeResize(U.rows(), U.cols() + 1); 
			U.col(U.cols() - 1) = u; 

		if (W.cols() == 0)
			W = w;
		else
			W.conservativeResize(W.rows(), W.cols() + 1); 
			W.col(W.cols() - 1) = w; 
#endif
		if (C.cols() == 0)
			C = c;
		else
			C.conservativeResize(C.rows(), C.cols() + 1); 
			C.col(C.cols() - 1) = c; 

		float temp = t.norm();
		p = X.transpose() * t / (temp * temp);
#if (0)
		temp = u.norm();
		q = Y.transpose() * u / (temp * temp);
#endif
		if (P.cols() == 0)
			P = p;
		else
			P.conservativeResize(P.rows(), P.cols() + 1); 
			P.col(P.cols() - 1) = p; 
#if (0)
		if (Q.cols() == 0)
			Q = q;
		else
			Q.conservativeResize(Q.rows(), Q.cols() + 1); 
			Q.col(Q.cols() - 1) = q; 
#endif
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
MatrixXf PLS::pseudoMat(MatrixXf P)
{
	MatrixXf P_transpose = P.transpose();
	MatrixXf pseudoM = P_transpose.completeOrthogonalDecomposition().pseudoInverse();
	return pseudoM;
}


const MatrixXf &PLS::getB() const
{
	return this->B;
}


const MatrixXf &PLS::getMeanX() const
{
	return this->mean0X;
}


const MatrixXf &PLS::getMeanY() const
{
	return this->mean0Y;
}


MatrixXf PLS::predict(
	const MatrixXf &v ) const
{
	MatrixXf temp;
	MatrixXf result = MatrixXf::Zero(v.rows(),B.cols());	
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
