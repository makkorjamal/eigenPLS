#ifndef PLS_HH
#define PLS_HH


#include <Eigen/Dense>
using Eigen::MatrixXf;


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
		MatrixXf B, mean0X, mean0Y;
		inline void display(
			const char *name,
			const MatrixXf &value );

	public:
		PLS( );
		PLS( const MatrixXf &B, const MatrixXf &meanX, const MatrixXf &meanY );
		~PLS();

		void train(
			const MatrixXf &Xdata,
			const MatrixXf &Ydata,
			float epsilon = 0.0001 );

		const MatrixXf &getB() const;

		const MatrixXf &getMeanX() const;

		const MatrixXf &getMeanY() const;

		MatrixXf predict(
			const MatrixXf &v ) const;

		MatrixXf pseudoMat(
				MatrixXf P);

};

#endif // PLS_HH
