N=10^5; threshold =15;
n=50; p=0.4; X_0=0; q=0.65;
X_50=zeros(N,1); LRs = zeros(N,1);

for i=1:N
    X=X_0; LR=1;
    for j=1:n
        Y = rand <= q;
        LR = LR * (p/q * (Y==1) + (1-p)/(1-q)* (Y==0));
        X=X+2*Y - 1;
    end
    LRs(i) = LR;
    X_50(i) = X;
end
ell_est = mean(LRs .* (X_50 > threshold))
RE_est = std(LRs .* (X_50 > threshold)) / (sqrt(N) * ell_est)