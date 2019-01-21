function [W FI H F_obj]=AnsNMF(V,r,eps,maxiter,deltaH,deltaW,W0,H0,S,flagW)
% This code is shared for research only. Please cite the following paper: 
%Z. Yang, Y. Xiang, K. Xie, and Y. Lai, "Adaptive method for nonsmooth nonnegative matrix factorization," IEEE Trans. Neural Networks and Learning Systems, vol. 28, no. 4, pp. 948-960, Apr. 2017.

delt=1e-12;
if flagW==0
    delt=0;
    
end
[n N]=size(V); % V contains your data in its column vectors


FI0=S;
km=maxiter-1;

iH=0;
iW=0;
 F_obj(1,1)=sum(sum((V-W0*FI0*H0).*(V-W0*FI0*H0)))/sum(sum(V.*V));

for iter=1:maxiter

 W1=W0*FI0;
    H0=(H0+delt).*(W1'*V+eps)./(W1'*W1*H0+eps);   
    Ae=eye(r,r);
    if mod(iter,km)==1
        iH=iH+1;
        deltaH=deltaH/iH^10;
        [y0 y1 obj0]=FNCIVM_H(H0,deltaH); %y0=y1*H0
        Ae=max(inv(y1),0);
        H0=max(y0,0);
    end

    
    
    FI0=FI0*Ae;

    H1=FI0*H0; 
    W0=(W0+delt).*(V*H1'+eps)./(W0*H1*H1'+eps);

    FI00=eye(r,r);
    if mod(iter,km)==1&&flagW==1
        
        iW=iW+1;
        deltaW=deltaW/iW;
        [FI00 W00 obj0]=FNCIVM(W0',deltaW);
        FI00=max(FI00,0);
        W0=max(W00',0);
    end
    
 W0=W0*diag(1./sum(W0,1)); 
    FI0=FI00'*FI0;    
    W=W0;    
    H=H0;
    FI=FI0;
    F_obj(1,iter+1)=sum(sum((V-W*FI*H).*(V-W*FI*H)))/sum(sum(V.*V));

end

function [out1 out2 obj]=FNCIVM(X0,delta)

% out1: the estimated mixing matrix   row sum-to-one
% out2: the estimated sources
% obj: the value of the objective function in each iteration
% addpath SeDuMi_1_3;

X0=diag(1./sum(X0,2))*X0;
[mm0 nn0]=size(X0);
if mm0<3
    [out1 out2 obj]=nLCA_IVM2(X0);
else
X1=X0';

X=X0;
[m0 n0]=size(X);

W=eye(mm0,mm0);
m=mm0;
v=abs(det(W));
j00=1;
k00=1;
for iter=1:m
    
    A_P=adj_mat(W);
    i=mod(iter,m);
    if i==0 i=m; end
    for j=1:m
        temp1(:,:)=A_P(i,j,:,:);
        obj_coef(j)=(-1)^(i+j)*det(temp1);
    end
    k0=0;
    
    for k=1:m
         if k<i   q_i=i-1; k0=k0+1;    end
         if k>i   q_i=i;   k0=k0+1;  end
         if k==i continue; end
         for j=1:m
            ineq_con_coef(k0,j,:)=zeros(1,m);
            ind3=[1:j-1 j+1:m];
            temp2(:,:)=A_P(k,j,:,:);
            ind_q1=[1:q_i-1 q_i+1:m-1];
            for q_j=1:m-1
                ind_q2=[1:q_j-1 q_j+1:m-1];
                temp_coef(1,q_j)=(-1)^(k+j)*(-1)^(q_i+q_j)*det(temp2(ind_q1,ind_q2));
            end
            ineq_con_coef(k0,j,ind3)=temp_coef;
        end
    end
    A1=[];
    for tt1=1:m-1
        for tt2=1:m
            ta(1,:)=ineq_con_coef(tt1,tt2,:);
            A1=[A1;-ta];
        end
    end
    
%     A10=-A1;
    b1=zeros(length(A1(:,1)),1);
    A2=-X';
    b2=zeros(n0,1)+delta;
%     A01=[A10;A2];
    A02=[A1;A2];
    b=[b1;b2];
    Aeq=ones(1,m0);
    beq=1;
    f=obj_coef';
    [y2,fval2,exitflag2]=linprog(-f,A02,b,Aeq,beq,[],[],W(i,:)',optimset('Display','off'));
    obj0(1,k00)=exitflag2; 
     k00=k00+1;     
    if exitflag2==1
        W(i,:)=y2';
    end
    
   
    p=f'*W(i,:)'; 
    obj01(j00)=v; j00=j00+1; v=p;     
end
out1=inv(W);
out2=W*X0;
obj=obj01;
end

function [A S obj]=nLCA_IVM2(X0)
obj=1;
X0=diag(1./sum(X0,2))*X0;
[mm0 nn0]=size(X0);
temp_X=X0(1,:)-X0(2,:);
ind1=find(temp_X>0);
ind2=find(temp_X<0);
X_ind1=X0(:,ind1);
X_ind2=X0(:,ind2);
temp1=-X_ind1(2,:)./temp_X(ind1);
temp2=-X_ind2(2,:)./temp_X(ind2);
W11=max(temp1);
W21=min(temp2);
W12=1-W11;
W22=1-W21;
WW=[W11 W12; W21 W22];
A=inv(WW);
S=WW*X0;

function [y0 W obj]=FNCIVM_H(X0,delta)
%mca, constrain the nonnegativity of the mixing matrix
% delta=-1e-1;
% delta=0;
% epss=1e-1*max(sum(X0)); %epss=1e-6;
epss=1e-6;
ind0=find(sum(X0)>epss);
X=X0(:,ind0);
[m N]=size(X);
X=X./(ones(m,1)*sum(X));

if m==2
    W=IVM_MSC2(X);
    y0=W*X0;
    obj=1;
else
% [A_est, location, yy] = VCA(X,'Endmembers',m,'SNR',30,'verbose','off');
% A_est=A_est./(ones(m,1)*sum(A_est));
% beta=0.01;
% A_est=vol_trans(A_est,X,beta);
% W=inv(A_est);

W=eye(m,m);
obj(1)=abs(det(W));
k0=1;
j0=1;
ind=1:m;
Iteration_max=m;
iter_cnt=0;
TOL_convergence=1e-12;
rec=100;
i=1;
while (iter_cnt<Iteration_max) %&(rec>TOL_convergence)
    W0=W';
    A_P=adj_mat(W0);
    i=mod(iter_cnt,m);
    if i==0 i=m; end
    for j=1:m
        temp1(:,:)=A_P(i,j,:,:);
        obj_coef(j)=(-1)^(i+j)*det(temp1);
    end
    k0=0;
    
    for k=1:m
         if k<i   q_i=i-1; k0=k0+1;    end
         if k>i   q_i=i;   k0=k0+1;  end
         if k==i continue; end
         for j=1:m
            ineq_con_coef(k0,j,:)=zeros(1,m);
            ind3=[1:j-1 j+1:m];
            temp2(:,:)=A_P(k,j,:,:);
            ind_q1=[1:q_i-1 q_i+1:m-1];
            for q_j=1:m-1
                ind_q2=[1:q_j-1 q_j+1:m-1];
                temp_coef(1,q_j)=(-1)^(k+j)*(-1)^(q_i+q_j)*det(temp2(ind_q1,ind_q2));
            end
            ineq_con_coef(k0,j,ind3)=temp_coef;
        end
    end
    A1=[];
    for tt1=1:m-1
        for tt2=1:m
            ta(1,:)=ineq_con_coef(tt1,tt2,:);
            A1=[A1;-ta];
        end
    end
    
    A10=A1;
    b1=zeros(length(A1(:,1)),1);
    
    
%     W0=W;
    for kj=1:m
        ss(kj,:,:)=W(:,kj)*X(kj,:);
    end
    for i1=1:m
        ind1=find(ind<i1|ind>i1);
        for i2=1:m
            ind2=find(ind<i2|ind>i2);
            temp_W=W(ind2,ind1);
            f0(i2,i1)=(-1)^(i1+i2)*det(temp_W);
        end
    end
   ss0=zeros(m,N);   
   if mod(i,m)>0
    kk=mod(i,m);
%         kk=t; 
        ind3=find(ind<kk|ind>kk);
         f=f0(:,kk); 
         ss_temp=ss(ind3,:,:);
         for kj=1:m-1
             ss01(:,:)=ss_temp(kj,:,:);
             ss0=ss0+ss01;
         end        
        ss0=ss0./(ones(m,1)*max(X(kk,:),epss));
        
        for j=1:m
            m_min(j,1)=min(ss0(j,:));
            m_max(j,1)=max(ss0(j,:));
        end
    end
   
    if mod(i,m)==0
        kk=m; 
        ind3=find(ind<kk|ind>kk);
         f=f0(:,kk); 
         ss_temp=ss(ind3,:,:);
         for kj=1:m-1
             ss01(:,:)=ss_temp(kj,:,:);
             ss0=ss0+ss01;
         end        
       ss0=ss0./(ones(m,1)*max(X(kk,:),epss));
        for j=1:m
            m_min(j,1)=min(ss0(j,:));
            m_max(j,1)=max(ss0(j,:));
        end
    end

    Aeq=ones(1,m);
    beq=1;
%     A=[ones(1,m);-ones(1,m)];
%     b=[1+delt delt-1];
    LB=-m_min-delta;
    A11=-eye(m,m);
    b11=-LB;
    A=[A10;A11];
    b=[b1;b11];

    [y2,fval2,exitflag2]=linprog(-f,A,b,Aeq,beq,[],[],W(:,kk),optimset('Display','off'));

    
  if exitflag2==1
        W(:,kk)=y2;
    end

%     if kk==m 
%         obj(j0)=abs((v-max(abs(p),abs(q))))/v; 
        
        j0=j0+1; 
%         v=max(abs(p),abs(q)); 
        obj(j0)=abs(det(W));    
%         rec = abs(abs(det(W0))-abs(det(W)))/abs(det(W0));
        iter_cnt=iter_cnt+1;
        i=i+1;
%     end
    end
% rec;
% iter_cnt;


y0=W*X0;
% figure
% subplot(2,1,1)
% plot(obj0(1,:))
% subplot(2,1,2)
% plot(obj0(2,:))
end


function A_P=adj_mat(A)
[m n]=size(A);
if m>n||m<n
    fprintf('the matrix is not square\n');
    return;
end
for i=1:m
    ind1=[1:i-1 i+1:m];
    for j=1:m
        ind2=[1:j-1 j+1:m];
        A_P(i,j,:,:)=A(ind1,ind2);
    end
end

function A_est=vol_trans(A,X,beta)
% beta is a slack variable which is no less than 0 

[p pp]=size(A);
s=inv(A)*X;

% beta=0.01;
ind=1:p;
for i=1:p
    IA=inv(A);
    ind1=find(ind<i|ind>i);
    AA=A(:,ind1);
    [jj kj]=min(s(i,:));
    delt=(1-IA(i,:))*X(:,kj)+beta-1;
    for j=1:p-1
        AA(:,j)=A(:,i)+(1+delt)*(AA(:,j)-A(:,i));
    end
    A(:,ind1)=AA;
    A_est=A;
%     inv(A_est)*X
end

function W=IVM_MSC2(X)
h=max(X(1,:));
f=min(X(1,:));
W(1,1)=(1-f)/(h-f);
W(1,2)=(-f)/(h-f);
W(2,:)=1-W(1,:);



