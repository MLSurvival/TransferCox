#include "head.hpp"
#include "coxnotied_strong.hpp"
#include "Coxl21loop_strong.hpp"

ColumnVector find_top_n(ColumnVector W, int n, double cindex){
	ColumnVector out(n);
	int count=0;
	double max=0.0;
	out(0)=cindex;
	for(int i=0;i<W.rows();i++){
		if(W(i)!=0.0) count +=1;
		if(W(i)<0.0) W(i)=(-1)*W(i);
	}
	out(1)=count;
	int k=2;
	while(k<n){
		out(k)=0;
		max=W(0);
		for(int i=1;i<W.rows();i++){
			if(W(i)>max){
				max=W(i);
				out(k)=i;
			}
		}
		W(out(k))=0.0;
		k +=1;
	}
	return out;
}

int main(int argc, char *argv[])
{
	cout<<"training1 file,training2 file,testing file, number of training1, number of training2,number of testing"<<endl;
	cout<<" number of feature, iteration,rho1,rho2,esp"<<endl;

    string nametrain=argv[1];
    string nametrain_T=argv[2];
    string nametest=argv[3];
    int num_train_rows=atoi(argv[4]); // Total number of train instances in the dataset.
    int num_train_rows2=atoi(argv[5]); // Total number of train instances in the dataset.
    int test_rows=atoi(argv[6]); // Total number of test instances in the dataset.
    int num_cols=atoi(argv[7]); // Total number of features in the dataset.
    int max_iter=atoi(argv[8]);//Variable for maximum number of iteration
    double weight=atof(argv[9]);  // step size (0.2)parameter.
    double rho2=atof(argv[10]);
    double eps=atof(argv[11]);
    int num_lambda=atoi(argv[12]);
    double step=atof(argv[13]);

    ColumnVector Betainit(num_cols);
    Betainit.setZero(num_cols);
    ColumnVector Betainit_T(num_cols);
    Betainit_T.setZero(num_cols);
    ColumnVector Betatemp_T(num_cols);
    Betatemp_T.setZero(num_cols);

    Cox21 newcox(nametrain,nametrain_T,nametest,num_train_rows,num_train_rows2,num_cols);
    Coxnotied test_part;
    test_part.create_from_file(nametest,test_rows,num_cols);

    time_t t1,t2;
	MatrixXd W(num_cols,2);
	MatrixXd Topfeatures(52,num_lambda);
	MatrixXd allrun(4,num_lambda);
	MatrixXd runinfor(56,num_lambda);
	MatrixXd Test_result(test_rows,(num_lambda+2));
	MatrixXd Train_T_result(num_train_rows2,(num_lambda+2));
	MatrixXd ALL_Beta(num_cols,num_lambda);
	Test_result.col(0)=test_part.Times;
	Test_result.col(1)=test_part.Status;
	Train_T_result.col(0)=newcox.train_part_T.Times;
	Train_T_result.col(1)=newcox.train_part_T.Status;

	double lambda_max=newcox.find_lambda_max(rho2,weight);


	double cindex=0.0;
	double lambda=0.0;
	double lambda_old=0.0;
	t1=time(0);
	for(int i=0;i<(num_lambda);i++){
		lambda_old=lambda_max*pow (step, double(i)/double(num_lambda));
		lambda=lambda_max*pow (step, double(i+1)/double(num_lambda));
		W=newcox.cox_L21_strong (Betainit, Betainit_T, rho2,lambda, lambda_old,max_iter,eps,weight);
		Betainit=W.col(0);
		Betainit_T=W.col(1);
		ALL_Beta.col(i)=W.col(1);
		allrun(0,i)=lambda;
		allrun(1,i)=newcox.obj;
		allrun(2,i)=newcox.funcv2;
		allrun(3,i)=newcox.featureleft;
		cout<<i<<endl;
	}
	t2=time(0);
	float diff((float)t2-(float)t1);
	cout<<"running time is"<<diff<<endl;

	for(int i=0;i<num_lambda;i++){
		Betatemp_T=ALL_Beta.col(i);
		// record the estimated value for BS in R
		test_part.estimate(Betatemp_T);
		Test_result.col((i+2))=test_part.predict;
		newcox.train_part_T.estimate(Betatemp_T);
		Train_T_result.col((i+2))=newcox.train_part_T.predict;
		// recorde the cindex, objective, nonzero, and top 50 features.
		cindex=test_part.GetCindex_yan();
		Topfeatures.col(i)=find_top_n(Betatemp_T, 52, cindex);
		//double nonzero=Betainit_T.nonZeros();
	}
	runinfor<<allrun,
			Topfeatures;
  //****************************************************************************************************************************//
	//                                     Prepare the output file at here
	string name = argv[1];
	string out=boost::lexical_cast<string>(step);
	string outname1=name+out+"_train_result_new.txt";
	string outname2=name+out+"_test_result_new.txt";
	string outname3=name+out+"_record_new.txt";
	string outname4=name+out+"_runtime.txt";
	const char *outputname1=outname1.c_str();
	const char *outputname2=outname2.c_str();
	const char *outputname3=outname3.c_str();
	const char *outputname4=outname4.c_str();
	ofstream outfile1,outfile2,outfile3,outfile4;
	outfile1.open(outputname1);
	outfile2.open(outputname2);
	outfile3.open(outputname3);
	outfile4.open(outputname4);
	outfile1<<Train_T_result<<endl;
	outfile2<<Test_result<<endl;
	outfile3<<runinfor<<endl;
	outfile4<<"running time is "<<diff<<endl;
	//cout<<runinfor<<endl;
	outfile1.close();
	outfile2.close();
	outfile3.close();
	outfile4.close();
	return 0;
}
