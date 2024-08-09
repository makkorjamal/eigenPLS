#ifndef PLS_HH
#define PLS_HH


#include <Eigen/Dense>
using Eigen::MatrixXd;


/**
 * Class to train and project PLS models.
 *
 * This implementation is based on the python PLS implementation by Avinash Kak (kak@purdue.edu).
 * Both implementations are based on the description of the algorithm by Herve Abdi in
 * the article "Partial Least Squares Regression and Projection on Latent Structure
 * Regression," Computational Statistics, 2010.
 */
class PLS
{
	private:
		MatrixXd B, mean0X, mean0Y;
		inline void display(
			const char *name,
			const MatrixXd &value );

	public:
		PLS( );
		PLS( const MatrixXd &B, const MatrixXd &meanX, const MatrixXd &meanY );
		~PLS();

		void train(
			const MatrixXd &Xdata,
			const MatrixXd &Ydata,
			double epsilon = 0.0001 );

		const MatrixXd &getB() const;

		const MatrixXd &getMeanX() const;

		const MatrixXd &getMeanY() const;

		MatrixXd predict(
			const MatrixXd &v ) const;

		MatrixXd pseudoMat(
				MatrixXd P);

};

#endif // PLS_HH
