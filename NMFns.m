function [W FI H F_obj]=NMFns(V,r,eps,maxiter,W0,H0,S)
[n N]=size(V);
W=W0;
H=H0;
FI=S;
F_obj(1,1)=sum(sum((V-W*FI*H).*(V-W*FI*H)))/sum(sum(V.*V));
for iter=1:maxiter
    H=H.*((W*S)'*V+eps)./((W*S)'*(W*S)*H+eps);
    W=W.*(V*(S*H)'+eps)./(W*(S*H)*(S*H)'+eps);
    W=W*diag(1./sum(W,1));
    F_obj(1,iter+1)=sum(sum((V-W*FI*H).*(V-W*FI*H)))/sum(sum(V.*V));
end


