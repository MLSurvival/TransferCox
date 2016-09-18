

#ifndef HEAD_HPP_
#define HEAD_HPP_

#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include <string.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include <cstdlib>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace boost;
using namespace Eigen;

typedef Eigen::Matrix<double, Dynamic, 1> ColumnVector;
typedef Eigen::Matrix<double, 1, Dynamic> RowVector;
typedef Eigen::Matrix<double, Dynamic, Dynamic> MatrixXd;

double pall_dot(RowVector feature, ColumnVector beta){
	double sum=0.0; int nf=beta.rows();
	omp_set_num_threads(36);	
	#pragma omp parallel
	{
		#pragma omp for reduction(+:sum)
		for(int i=0;i<nf;i++)
			sum +=feature[i]*beta[i];
	}
		return sum;
}

#endif /* HEAD_HPP_ */
