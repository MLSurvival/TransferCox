
#ifndef COXNOTIED_HPP_
#define COXNOTIED_HPP_

#include "head.hpp"

class Coxnotied{
public:
	int num_rows,num_cols,num_cols_all;
	MatrixXd Features_org; /*Matrix representation of the Features dataset*/
	MatrixXd Features; /*Matrix representation of the Features dataset*/
	ColumnVector Status; /*Column representation of the Status attribute*/
	ColumnVector Times; /*Column representation of the Times attribute*/
	void create_from_file(string,int,int);
	void prepare();
	void updatefeature(ColumnVector);
	double ComputeLogLikelihood(ColumnVector);
	ColumnVector calculate_gradient_fast(ColumnVector,double,double);
	ColumnVector calculate_gradient_fast_allfeature(ColumnVector ,double ,double );
	void estimate(ColumnVector);
	double GetCindex_yan();
	ColumnVector predict;
private:
	int num_inde_time;
	vector<double> SortedUniqueFailureTimes;
	map<int, vector<int> > Ri;
	map<int, vector<int> > Di;
	void ComputeUniqueTimes();
	void ComputeRiDi();
	void groupDifeature();
	void groupDifeatureall();
	MatrixXd Groupfeature;
	MatrixXd Groupfeatureall;
};

void Coxnotied::create_from_file(string filename,int r, int c){
	num_rows=r;
	num_cols_all=c;
	Features_org.setZero(num_rows,num_cols_all);
	Status.setZero(num_rows);
	Times.setZero(num_rows);
	int rowcount=0;
	int colindex=0;
	string line;
	const char *ffname=filename.c_str();
	 vector <string> fields;
	  ifstream myfile1;
	    myfile1.open(ffname);

	    if(myfile1.is_open())
	    {
	        while (!myfile1.eof())
	        {
	            getline(myfile1,line);
	            split( fields, line, is_any_of( "," ) );
	            if(fields.size()>1) //A condition added here to avoid segmentation fault!
	            {
	                vector<string>::iterator iter;
	                colindex=0;
	                for(int i=0;i<fields.size()-2;i++)
	                {
	                	Features_org(rowcount,colindex)=atof(fields[i].c_str()); //Copying the attributes to the matrix
	                    colindex++;
	                }
	                Times(rowcount)=atof(fields[fields.size()-2].c_str());
	                Status(rowcount)=atoi(fields[fields.size()-1].c_str());
	            }
	            rowcount++;
	        }
	    }
	    myfile1.close();
	    predict.setZero(num_rows);
}

void Coxnotied::ComputeUniqueTimes()
{
	for(int i=0;i<Times.rows();i++){
		if(Status[i]==1){
		SortedUniqueFailureTimes.push_back(Times[i]);}
	}

	sort(SortedUniqueFailureTimes.begin(), SortedUniqueFailureTimes.end()); //sort in asecending order.
    SortedUniqueFailureTimes.erase(unique(SortedUniqueFailureTimes.begin(), SortedUniqueFailureTimes.end()), SortedUniqueFailureTimes.end());
}

void Coxnotied::ComputeRiDi()
{
	num_inde_time=SortedUniqueFailureTimes.size();
	for(int i=0;i<SortedUniqueFailureTimes.size();i++)
	{
		for(int j=0;j<Times.rows();j++)
		{
			if(i==0){
				if(Times[j]>=SortedUniqueFailureTimes[i])
					Ri[i].push_back(j);
			}else{
				if((Times[j]<SortedUniqueFailureTimes[i])&&(Times[j]>=SortedUniqueFailureTimes[i-1]))
					Ri[i].push_back(j);
			}
			if((Times[j]==SortedUniqueFailureTimes[i])&&(Status[j]==1))
			    Di[SortedUniqueFailureTimes[i]].push_back(j);
		}
	}
}

void Coxnotied::groupDifeature(){
	//for(int i=id;i<num_inde_time;i=i+nthreads)
		for(int i=0;i<num_inde_time;i++){
		vector<int> Indices;
		RowVector Feature(num_cols);
		Indices=Di[SortedUniqueFailureTimes[i]];
		Feature.setZero(num_cols);
		for(int k=0;k<Indices.size();k++)
			Feature=Feature + Features.row(Indices[k]);
		Groupfeature.row(i)=Feature;
	    }
}

void Coxnotied::groupDifeatureall(){
	Groupfeatureall.setZero(num_inde_time,num_cols_all);
	//for(int i=id;i<num_inde_time;i=i+nthreads)
		for(int i=0;i<num_inde_time;i++){
		vector<int> Indices;
		RowVector Feature(num_cols_all);
		Indices=Di[SortedUniqueFailureTimes[i]];
		Feature.setZero(num_cols_all);
		for(int k=0;k<Indices.size();k++)
			Feature=Feature + Features_org.row(Indices[k]);
		Groupfeatureall.row(i)=Feature;
	    }
}

void Coxnotied::prepare(){
	ComputeUniqueTimes();
	ComputeRiDi();
	groupDifeatureall();
}

void Coxnotied::updatefeature(ColumnVector indicator){
	num_cols=(int)indicator.sum();
	Features.setZero(num_rows,num_cols);
	int k=0;
	for(int i=0;i<indicator.rows();i++){
		if(indicator(i)==1){
			Features.col(k)=Features_org.col(i);
			k +=1;
		}
	}
	Groupfeature.setZero(num_inde_time,num_cols);
	groupDifeature();
}

double Coxnotied::ComputeLogLikelihood(ColumnVector Beta) // Function to compute the cox partial log likelihood value.
{
/*	double sum2=0.0;
	#pragma omp parallel
	{
	int id, nthreads;
	id=omp_get_thread_num();
	nthreads=omp_get_num_threads();*/

    double temp7; double temp8;
    double sum=0.0;
    vector<int>::iterator iter1;
    //for(int i=id;i<num_inde_time;i=i+nthreads)
    double temp4; double temp5; double temp2=0.0;
		for(iter1=Ri[0].begin();iter1!=Ri[0].end();iter1++)
		{
			temp4=pall_dot(Features.row(*iter1), Beta);
			//temp4=(Features.row(*iter1)).dot(Beta);
			temp5=exp(temp4);
			temp2=temp2 + (temp5);
		}
		temp8= Di[SortedUniqueFailureTimes[0]].size()*log(temp2);
		temp7= pall_dot(Groupfeature.row(0), Beta);
		//temp7= (Groupfeature.row(0)).dot(Beta);
		if(temp8!=0)
			sum=temp7-temp8;

		for(int i=1;i<num_inde_time;i++)
		{
			for(iter1=Ri[i].begin();iter1!=Ri[i].end();iter1++)
			{
				temp4=pall_dot(Features.row(*iter1), Beta);
				//temp4=(Features.row(*iter1)).dot(Beta);
				temp5=exp(temp4);
				temp2=temp2 - (temp5);
			}
			temp8= Di[SortedUniqueFailureTimes[i]].size()*log(temp2);

			temp7= pall_dot(Groupfeature.row(i), Beta);
			//temp7= (Groupfeature.row(i)).dot(Beta);

			if(temp8!=0)
				sum=sum + (temp7-temp8);
		}
//	#pragma omp critical
//			sum2 +=sum;
//	}
    return (sum/num_rows);
}

ColumnVector Coxnotied::calculate_gradient_fast(ColumnVector Beta,double rho_L2,double weight)
//function to calculate the negative gradient.
{
	//clock_t t1,t2;
	//t1=clock();
    ColumnVector Gradient;
    Gradient.setZero(num_cols);
//    RowVector Gradient_trans2;
 //   Gradient_trans2.setZero(num_cols);

//#pragma omp parallel
//	{
//    int id, nthreads;
 //   id=omp_get_thread_num();
 //   nthreads=omp_get_num_threads();
        vector <int>::iterator iter1;
		RowVector Ri_Xi(num_cols);
		double temp4,temp5,temp6;
		RowVector Gradient_trans;
		Gradient_trans.setZero(num_cols);
		temp6=0.0;
		Ri_Xi.setZero(num_cols);
		for(iter1=Ri[0].begin();iter1!=Ri[0].end();iter1++)
		{
			temp4=pall_dot(Features.row(*iter1), Beta);
			//temp4=(Features.row(*iter1)).dot(Beta);
			temp5=exp(temp4);
			Ri_Xi=Ri_Xi + (Features.row(*iter1))*temp5;
			temp6=temp6 + temp5;
		}
		Gradient_trans=(-1*Groupfeature.row(0)  + (Di[SortedUniqueFailureTimes[0]].size()*Ri_Xi)/temp6);


		for(int i=1;i<num_inde_time;i++)
		{
            for(iter1=Ri[i].begin();iter1!=Ri[i].end();iter1++)
            {
                temp4=pall_dot(Features.row(*iter1), Beta);
            	//temp4=(Features.row(*iter1)).dot(Beta);
                temp5=exp(temp4);
                Ri_Xi=Ri_Xi - (Features.row(*iter1))*temp5;
                temp6=temp6 - temp5;
            }
            Gradient_trans=Gradient_trans + (-1*Groupfeature.row(i)  + (Di[SortedUniqueFailureTimes[i]].size()*Ri_Xi)/temp6);
		}
//#pragma omp critical
//        Gradient_trans2 +=Gradient_trans;

//	}
	for(int j=0;j<num_cols;j++)
        Gradient[j]=Gradient_trans[j]/num_rows*weight+2*rho_L2*Beta[j];

    //t2=clock();
    //float diff((float)t2-(float)t1);
    // cout<<diff<<endl;
    //cout<<"the gradient of cox is"<<endl;
    //cout<<Gradient<<endl;
    return Gradient;
}



ColumnVector Coxnotied::calculate_gradient_fast_allfeature(ColumnVector Beta,double rho_L2,double weight)
//function to calculate the negative gradient.
{
	//clock_t t1,t2;
	//t1=clock();
    ColumnVector Gradient;
    Gradient.setZero(num_cols_all);
//    RowVector Gradient_trans2;
 //   Gradient_trans2.setZero(num_cols);

//#pragma omp parallel
//	{
//    int id, nthreads;
 //   id=omp_get_thread_num();
 //   nthreads=omp_get_num_threads();
        vector <int>::iterator iter1;
		RowVector Ri_Xi(num_cols_all);
		double temp4,temp5,temp6;
		RowVector Gradient_trans;
		Gradient_trans.setZero(num_cols_all);
		temp6=0.0;
		Ri_Xi.setZero(num_cols_all);
		for(iter1=Ri[0].begin();iter1!=Ri[0].end();iter1++)
		{
			temp4=pall_dot(Features_org.row(*iter1), Beta);
			//temp4=(Features.row(*iter1)).dot(Beta);
			temp5=exp(temp4);
			Ri_Xi=Ri_Xi + (Features_org.row(*iter1))*temp5;
			temp6=temp6 + temp5;
		}
		Gradient_trans=(-1*Groupfeatureall.row(0)  + (Di[SortedUniqueFailureTimes[0]].size()*Ri_Xi)/temp6);


		for(int i=1;i<num_inde_time;i++)
		{
            for(iter1=Ri[i].begin();iter1!=Ri[i].end();iter1++)
            {
                temp4=pall_dot(Features_org.row(*iter1), Beta);
            	//temp4=(Features.row(*iter1)).dot(Beta);
                temp5=exp(temp4);
                Ri_Xi=Ri_Xi - (Features_org.row(*iter1))*temp5;
                temp6=temp6 - temp5;
            }
            Gradient_trans=Gradient_trans + (-1*Groupfeatureall.row(i)  + (Di[SortedUniqueFailureTimes[i]].size()*Ri_Xi)/temp6);
		}
//#pragma omp critical
//        Gradient_trans2 +=Gradient_trans;

//	}
	for(int j=0;j<num_cols_all;j++)
        Gradient[j]=Gradient_trans[j]/num_rows*weight+2*rho_L2*Beta[j];

    //t2=clock();
    //float diff((float)t2-(float)t1);
    // cout<<diff<<endl;
    //cout<<"the gradient of cox is"<<endl;
    //cout<<Gradient<<endl;
    return Gradient;
}



double Coxnotied::GetCindex_yan(){
	double cindex;
	int sum1=0; int sum2=0; int sum3=0; int sum4=0;
#pragma omp parallel
	{
		double stime1; double stime2; double pred1,pred2; int status1,status2;
#pragma omp for reduction(+:sum1,sum2,sum3,sum4)
		for(int i=0;i<num_rows-1;i++)
		{
			for(int j=i+1;j<num_rows;j++)
			{
				stime1=Times[i];
				stime2=Times[j];
				pred1= predict[i];
				pred2= predict[j];
				status1=Status[i];
				status2=Status[j];
				if(stime1 < stime2 && pred1 > pred2 && status1==1)
                    sum1=sum1 + 1;
				if(stime2 < stime1 && pred2 > pred1 && status2==1)
                    sum2=sum2 + 1;
				if(stime1 < stime2 && status1==1)
                    sum3=sum3 + 1;
				if(stime2 < stime1 && status2==1)
                    sum4=sum4 + 1;
			}
		}
	}
	cindex=(sum1 + sum2)/((sum3 + sum4)*1.0);
	//cout<<(sum1 + sum2)<<" "<<(sum3 + sum4)<<endl;
	return cindex;
}

void Coxnotied::estimate(ColumnVector Beta){
	for(int i=0;i<num_rows;i++){
		predict[i]=pall_dot(Features_org.row(i), Beta);
		//predict[i]=(Features.row(i)).dot(Beta);
	}
}
#endif /* COXNOTIED_HPP_ */
