/*
 * Coxl21loop.hpp
 *
 */

#ifndef COXL21LOOP_HPP_
#define COXL21LOOP_HPP_
#include "head.hpp"
#include "coxnotied_strong.hpp"
class Cox21 {
public:
	int num_train_rows,num_train_rows2,num_cols,num_cols_all;
    Coxnotied train_part_S;
    Coxnotied train_part_T;
    Cox21(string,string ,string ,int ,int ,int);
    MatrixXd cox_L21_mainloop (ColumnVector, ColumnVector, double,double ,int ,double ,double);
    double find_lambda_max( double ,double );
    void strongrule(ColumnVector , ColumnVector , double ,double ,double ,double );
    MatrixXd cox_L21_strong (ColumnVector , ColumnVector , double ,double ,double ,int ,double ,double );
    double obj;
    double funcv2;
    int featureleft;
private:
    int numtask;
    MatrixXd FGLasso_projection(MatrixXd ,double);
    double nonsmooth(MatrixXd ,double );
    double funcval_smooth(MatrixXd ,double ,double);
    bool strongrule_check(ColumnVector , ColumnVector , double ,double ,double );
    ColumnVector indicator;
};

Cox21::Cox21(string nametrain,string nametrain_T,string nametest,int r1,int r2,int c){
	num_train_rows=r1;
	num_train_rows2=r2;
	num_cols_all=c;
	train_part_S.create_from_file(nametrain,num_train_rows,num_cols_all);
	train_part_T.create_from_file(nametrain_T,num_train_rows2,num_cols_all);
	train_part_S.prepare();
	train_part_T.prepare();
	indicator.setOnes(num_cols_all);
	numtask=2;
	obj=0.0;
	funcv2=0.0;
	featureleft=0;
}

MatrixXd Cox21::FGLasso_projection(MatrixXd W,double lambda){
    MatrixXd Wp;
    Wp.setZero(num_cols,numtask);
    RowVector V;
    V.setZero(numtask);
    double nm=0.0,temp1=0.0;
    for(int i=0;i<num_cols;i++){
    	V=W.row(i);
    	nm=V.norm();
    	if(nm!=0.0){
    		temp1=((nm-lambda)>0)?(nm-lambda):0;
    		Wp.row(i)=V*(temp1/nm);
    	}
    }
    return Wp;
}

double Cox21::nonsmooth(MatrixXd W,double lambda){
	double sum=0.0;
    RowVector V;
    V.setZero(numtask);
    for(int i=0;i<num_cols;i++){
    	V=W.row(i);
    	sum +=V.norm()*lambda;
    }
    return sum;
}

double Cox21::funcval_smooth(MatrixXd temp_W,double rho_L2,double weight){
	double temp_s=0.0,temp_t=0.0,funcv=0.0;
	ColumnVector temp_WS(num_cols),temp_WT(num_cols);
	temp_WS=temp_W.col(0);
	temp_WT=temp_W.col(1);
	temp_s=train_part_S.ComputeLogLikelihood(temp_WS);
	temp_t=train_part_T.ComputeLogLikelihood(temp_WT);
	funcv=rho_L2*temp_W.squaredNorm()-temp_s-temp_t*weight;
	return funcv;
}

double Cox21::find_lambda_max( double rho_L2,double weight){
	ColumnVector g_WS(num_cols_all),g_WT(num_cols_all),Ws_S(num_cols_all),Ws_T(num_cols_all);
	Ws_S.setZero(num_cols_all);
	Ws_T.setZero(num_cols_all);
	g_WS= train_part_S.calculate_gradient_fast_allfeature(Ws_S,rho_L2,1.0);
	g_WT= train_part_T.calculate_gradient_fast_allfeature(Ws_T,rho_L2,weight);
	double lambda_max=0.0, temp=0.0;
	for(int i=0;i<num_cols_all;i++){
		temp=sqrt(g_WS(i)*g_WS(i)+g_WT(i)*g_WT(i));
		if(temp>lambda_max) lambda_max=temp;
	}
	return lambda_max;
}

void Cox21::strongrule(ColumnVector Beta, ColumnVector Beta_T, double rho_L2,double rho1,double rho_old,double weight){
	ColumnVector g_WS(num_cols_all),g_WT(num_cols_all);
	indicator.setOnes(num_cols_all);
	g_WS= train_part_S.calculate_gradient_fast_allfeature(Beta,rho_L2,1.0);
	g_WT= train_part_T.calculate_gradient_fast_allfeature(Beta_T,rho_L2,weight);
	double temp=0.0;
	for(int i=0;i<num_cols_all;i++){
		temp=sqrt(g_WS(i)*g_WS(i)+g_WT(i)*g_WT(i));
		if(temp<(2*rho1-rho_old))
			indicator(i)=0;
	}
	train_part_S.updatefeature(indicator);
	train_part_T.updatefeature(indicator);
	num_cols=(int)(indicator.sum());
	//cout<<num_cols<<endl;
}

bool Cox21::strongrule_check(ColumnVector Beta, ColumnVector Beta_T, double rho_L2,double rho1,double weight){
	ColumnVector g_WS(num_cols_all),g_WT(num_cols_all);
	g_WS= train_part_S.calculate_gradient_fast_allfeature(Beta,rho_L2,1.0);
	g_WT= train_part_T.calculate_gradient_fast_allfeature(Beta_T,rho_L2,weight);
	bool empty=true;
	double temp=0.0;
	for(int i=0;i<num_cols_all;i++){
		if(indicator(i)==0){
			temp=sqrt(g_WS(i)*g_WS(i)+g_WT(i)*g_WT(i));
			if(temp>(rho1)){
				indicator(i)=1;
				empty=false;
			}
		}
	}
	if(empty==false){
		train_part_S.updatefeature(indicator);
		train_part_T.updatefeature(indicator);
		num_cols=(int)(indicator.sum());
	}
	return empty;
}

MatrixXd Cox21::cox_L21_mainloop (ColumnVector Beta, ColumnVector Beta_T, double rho_L2,double rho1,int max_iter,double eps,double weight){

	double t=1.0, t_old=0.0, g=1.0,alpha=0.0,funcv1=0.0,funcv3=0.0,temp_lambda=0.0,r_sum=0.0,g_inc=2.0,object_old=0.0;
	int iter=0;
	bool bFlag=false;
	ColumnVector W_S(num_cols),W_S_old(num_cols),W_T(num_cols),W_T_old(num_cols);
	ColumnVector Ws_S(num_cols),Ws_T(num_cols),g_WS(num_cols),g_WT(num_cols);
	W_S=Beta;
	W_S_old=Beta;
	W_T=Beta_T;
	W_T_old=Beta_T;
	MatrixXd WTS(num_cols,2),gWTS(num_cols,2),temp_WGTS(num_cols,2),temp_W(num_cols,2),delta_W(num_cols,2);

	while(iter<max_iter){
		alpha=(t_old-1)/t;
		Ws_S=(1+alpha)*W_S-alpha*W_S_old;
		Ws_T=(1+alpha)*W_T-alpha*W_T_old;

		g_WS= train_part_S.calculate_gradient_fast(Ws_S,rho_L2,1.0);
		g_WT= train_part_T.calculate_gradient_fast(Ws_T,rho_L2,weight);
		gWTS<<g_WS,g_WT;
		WTS<<Ws_S,Ws_T;

		funcv1=funcval_smooth(WTS,rho_L2,weight);
		while(true){
			temp_lambda=rho1/g;
			temp_WGTS=WTS-gWTS/g;
			temp_W=FGLasso_projection(temp_WGTS,temp_lambda);
			funcv2=funcval_smooth(temp_W,rho_L2,weight);
			delta_W=temp_W-WTS;
			r_sum=delta_W.squaredNorm();
			funcv3=funcv1+g/2*r_sum+(delta_W.cwiseProduct(gWTS)).sum();
			if(r_sum<=0.000000000000000000001){
				bFlag=true;
				break;
			}
			if (funcv2<=funcv3) break;
			else g=g*g_inc;
		}
		W_S_old=W_S;
		W_S=temp_W.col(0);
		W_T_old=W_T;
		W_T=temp_W.col(1);

		object_old=obj;
		obj=funcv2+nonsmooth(temp_W,rho1);

		if(bFlag) {
			break;
		}
		if (iter>=2){
			if(abs(object_old-obj)<=eps)
				break;
		}

		iter++;
		t_old=t;
		t=0.5*(1+sqrt(1+4*t*t));
	}
	return temp_W;
}

MatrixXd Cox21::cox_L21_strong (ColumnVector Beta, ColumnVector Beta_T, double rho_L2,double rho1,double rho_old,int max_iter,double eps,double weight){
	strongrule(Beta,  Beta_T,  rho_L2, rho1, rho_old, weight);
	ColumnVector Beta_allS(num_cols_all),Beta_allT(num_cols_all);
	MatrixXd output(num_cols_all,2);
	Beta_allS=Beta;
	Beta_allT=Beta_T;
	bool empty=false;
	while(empty==false){
		ColumnVector Beta_up(num_cols), Beta_T_up(num_cols);
		MatrixXd sub_B(num_cols,2);
		int k=0;
		for(int i=0;i<num_cols_all;i++){
			if(indicator(i)==1){
				Beta_up(k)=Beta_allS(i);
				Beta_T_up(k)=Beta_allT(i);
				k +=1;
			}
		}
		sub_B=cox_L21_mainloop (Beta_up, Beta_T_up, rho_L2, rho1, max_iter, eps, weight);
		Beta_allS.setZero(num_cols_all);
		Beta_allT.setZero(num_cols_all);
		int kk=0;
		for(int i=0;i<num_cols_all;i++){
			if(indicator(i)==1){
				Beta_allS(i)=(sub_B.col(0))(kk);
				Beta_allT(i)=(sub_B.col(1))(kk);
				kk +=1;
			}
		}
		empty=strongrule_check(Beta_allS, Beta_allT, rho_L2,rho1,weight);
	}
	output<<Beta_allS,Beta_allT;
	featureleft=num_cols;
	return output;
}



#endif /* COXL21LOOP_HPP_ */
