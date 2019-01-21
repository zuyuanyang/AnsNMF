% This code is shared for research only. Please cite the following paper: 
%Z. Yang, Y. Xiang, K. Xie, and Y. Lai, "Adaptive method for nonsmooth nonnegative matrix factorization," IEEE Trans. Neural Networks and Learning Systems, vol. 28, no. 4, pp. 948-960, Apr. 2017.
clear
m=6;
r=3;
N=20;
eps=1e-9;
rand('state',1000)
W=rand(m,r);
rand('state',2000)
H=rand(r,N);
V=W*H;
c=max(max(V));
V=V/c;    %constructing V
maxiter=100;
rnum=50;  %maxiter and rnum could be smaller if the computing time is too long
flagW=1;
spa=[];
for rand_num=1:rnum
    rand('state',rand_num); 
    W01=rand(m,r); 
    W01=W01*diag(1./sum(W01,1));
    rand('state',rand_num*10);  
    H01=rand(r,N);
    theta=0.0;
    S=(1-theta)*eye(r,r)+theta/r*ones(r,1)*ones(1,r);
    [W0 FI0 H0 F_obj0]=NMFns(V,r,eps,10,W01,H01,S);  %initialization
    
    jH=0;
    for dh0=0.001:0.02:0.501
        jH=jH+1;
        jW=0;
        for dw0=0.001:0.02:0.501
            jW=jW+1;
            W=[];  FI=[]; H=[];F_obj=[];time=[];
            deltaH=max(max(H0))*dh0;
            deltaW=max(max(W0))*dw0;
            t0=cputime;
            [W FI H F_obj]=AnsNMF(V,r,eps,maxiter,deltaH,deltaW,W0,H0,FI0,flagW);
            time=cputime-t0;
            fname = ['results/sources' num2str(r) '_rand' num2str(rand_num) num2str(dh0+1) num2str(dw0+2) '.mat'];
            save(fname,'W','FI','H','F_obj','time');
            load(fname)
            WW=reshape(W,1,m*r);
            Ws=(sqrt(length(WW))-sum(abs(WW))/abs(sqrt(sum(WW.*WW))))/(sqrt(length(WW))-1);
            HH=reshape(H,1,r*N);
            Hs=(sqrt(length(HH))-sum(abs(HH))/abs(sqrt(sum(HH.*HH))))/(sqrt(length(HH))-1);
            spaH(rand_num,jH,jW)=Hs;
            spaW(rand_num,jH,jW)=Ws;
            FF(rand_num,jH,jW)=F_obj(1,end);
        end       
    end
end
save('smu_rand_para_HWF','spaH','spaW','FF')

%show fig. 1 and fig. 2 in the paper
load smu_rand_para_HWF
X=[0.001:0.02:0.501];
Y=[0.001:0.02:0.501];
jH=length(X);
jW=length(Y);
for i=1:jH
    for j=1:jW
        temp1=spaH(:,i,j);
        temp2=spaW(:,i,j);
        temp3=FF(:,i,j);
        HH0(i,j)=mean(temp1);
        WW0(i,j)=mean(temp2);
        EE0(i,j)=mean(temp3);
    end
end
figure
mesh(X,Y,HH0)
figure
mesh(X,Y,WW0)
figure
mesh(X,Y,EE0)



