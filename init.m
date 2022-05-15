load('yeast.mat');
[~,k] = size(data);
[c,n] = size(target);
k = k + 1;
A_init = normrnd(0,25,k,c);
C_init = normrnd(0,25,k,c);
B_init = normrnd(0,16,k,c);
D_init = normrnd(0,25,k,c);
E_init = normrnd(0,225/34,n,c);
