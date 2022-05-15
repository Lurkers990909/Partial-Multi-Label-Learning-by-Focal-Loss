load('A_init.mat');
load('B_init.mat');
load('C_init.mat');
load('D_init.mat');
load('E_init.mat');
load('yeast.mat');

intercept = ones(2417,1);
data = [data,intercept];

[~,k] = size(data);
[c,n] = size(target);

X_train = data;
Y_1  = transpose(target); 

alpha = 500;
beta = 1;
gamma =  0.1;
lambda_A = 1e-4;
lambda_C = 1e-4;
lambda_B = 1e-4; 
A = A_init;
B = B_init;
C = C_init;
D = D_init;
E = E_init;

% A = zeros(k,c);
% B = zeros(k,c);
% C = zeros(k,c);
% D = zeros(k,c);
% E = zeros(n,c);
% D = A + C;
% E = X_train * C;
theta = zeros(k,1);
eta = zeros(n,1);
mu_1 = 1;
mu_2 = 1;
rho = 1;
Q_1 = zeros(k,c);
Q_2 = zeros(n,c);
i = 1;
j = 1;
t = 1;
total_iter = 100;
internal_iter = 100; 

x_curve = zeros(1,internal_iter); % the iteration times
y_curve = zeros(1,internal_iter);
y_curve_for_D = zeros(1,total_iter);
y_curve_for_E = zeros(1,total_iter);

total_x_curve = zeros(1,total_iter);
total_y_curve = zeros(1,total_iter);
while (i<=internal_iter)
    x_curve(i) = i;
    i = i + 1;
end
i = 1;
 
tic
while(t <= total_iter)
    while(i <= 1) %Solve matrix A
        B_i = B(:,i);
        C_i = C(:,i);
        D_i = D(:,i);
        Q_1i = Q_1(:,i);
        theta = A(:,i);
        y_train_i = Y_1(:,i);
        
        while(j <= internal_iter)
            [value,grad] = costFunction_A(theta, X_train, y_train_i, B_i, C_i,D_i,Q_1i,mu_1);
            y_curve(j) = value;
            gap = lambda_A * grad;
            theta = theta - gap;
            j = j + 1;
        end
        j = 1;
        A(:,i) = theta;
        i = i + 1;
    end
    i = 1;
%     figure(1);%Draw the convergence curve
%     plot(x_curve,y_curve)
%     xlabel("iteration times");
%     ylabel("loss");
    
    
    
    while(i <= c) %Solve matrix C
        A_i = A(:,i);
        B_i = B(:,i);
        D_i = D(:,i);
        E_i = E(:,i);
        Q_1i = Q_1(:,i);
        Q_2i = Q_2(:,i);
        theta = C(:,i);
        y_train_i = Y_1(:,i);
        
        while(j <= internal_iter)
            [value,grad] = costFunction_C(theta, X_train, y_train_i, A_i, B_i,D_i,E_i,Q_1i,Q_2i,mu_1,mu_2);
            y_curve(j) = value;
            gap = lambda_C * grad;
            %             disp(gap)
            theta = theta - gap;
            j = j + 1;
        end
        j = 1;
        C(:,i) = theta;
        i = i + 1;
    end
%     figure(2);%Draw the convergence curve
%     plot(x_curve,y_curve)
%     xlabel("iteration times");
%     ylabel("loss");
    
    
    
    %Solve the matrix D

     
    [value,U,S,V] = costFunction_D(D,A, C,Q_1,mu_1,gamma);
    y_curve_for_D(t) = value;
    D = U * soft(S,gamma/mu_1) * V';
% %     M = A + C + Q_1/mu_1;
% %     D = SVT(M,gamma/mu_1,2);
    
    
    
    i = 1;
    while(i <= c) %Solve matrix B
        A_i = A(:,i);
        C_i = C(:,i);
        theta = B(:,i);
        y_train_i = Y_1(:,i);
        
        while(j <= internal_iter)
            [value,grad] = costFunction_B(theta, X_train, y_train_i, A_i, C_i, alpha);
            y_curve(j) = value;
            T = alpha * lambda_B / 2;
            theta = soft( theta - grad * lambda_B , T);
            j = j + 1;
        end
        j = 1;
        B(:,i) = theta;
        i = i + 1;
    end
%     figure(3);%Draw the convergence curve
%     plot(x_curve,y_curve)
%     xlabel("iteration times");
%     ylabel("loss");
    
    
    
   
%     Solve matrix E
    y_curve_for_E(t) = costFunction_E(E,X_train,C,Q_2,mu_2,beta);    
    optimal =  ( X_train * C + Q_2 / mu_2 );
    E = soft(optimal , beta/mu_2);


    
    
    
    
    
    
    
    i = 1;
    %Draw the total convergence curve
    total_x_curve(t) = t;
    total_y_curve(t) = total_loss(X_train,Y_1,A,B,C,D,E,Q_1,Q_2,mu_1,mu_2, alpha, beta, gamma,c);
    
    
    
    Q_1 = Q_1 + mu_1 .* (A + C - D);
    mu_1 = rho * mu_1;
    Q_2 = Q_2 + mu_2 .* (X_train * C - E);
    mu_2 = rho * mu_2;
    t = t + 1;
end
toc

figure(4);
plot(total_x_curve,total_y_curve)
xlabel("iteration times");
ylabel("loss");

% figure(2);
% plot(total_x_curve,y_curve_for_E)
% xlabel("iteration times");
% ylabel("loss");

% figure(3);
% plot(total_x_curve,y_curve_for_D)
% xlabel("iteration times");
% ylabel("loss");



disp(['运行时间: ',num2str(toc)]);


spa = length(find(B == 0));
disp(spa);
F = X_train * (A + B + C);
H = 1./(1+exp(-F));
[n,m] = size(Y_1);
z = 1;
while (z<=n*m)
        if H(z) >= 0.5
            H(z) = 1;
        else
            H(z) = 0;
        end
        z = z + 1;   
end
sim = (H == Y_1);
rate = length(find(sim == 1))/(n*m);
disp(rate);



i = 1;
t = 0;
while (i <= n*m)
    if(Y_1(i)~=H(i))
        t = t + 1;
    end
    i = i + 1;
end
loss = t/(n*m);
sprintf("The Hamming Loss is: %f",loss)

i = 1;
F_1 = 0;
while(i<=m)
    [score,~,~] = f1_score(Y_1(:,i),H(:,i));
    F_1 = F_1 + score;
    i = i + 1;
end
F_1 = F_1 / m;
sprintf("The F1 Score is: %f",F_1)
function [score, TPR, TNR] = f1_score(label, predict)
   M = confusionmat(label, predict);
   TPR = M(2,2) / (M(2,1) + M(2,2)); %SE: TP/(TP+FN)
   TNR = M(1,1) / (M(1,1) + M(1,2)); %SP: TN/(TN+FP)
   M = M';
   precision = diag(M)./(sum(M,2) + 0.0001);  %the sum of column: TP/(TP+FP)
   recall = diag(M)./(sum(M,1)+0.0001)'; %the sum of row: TP/(TP+FN)
   precision = mean(precision);
   recall = mean(recall);
   score = 2*precision*recall/(precision + recall);
end











%focal loss 
function [J,grad] = costFunction_A(theta, X, y,B_i,C_i,D_i,Q_1i,mu_1) % cost function for A
    m = length(y);
    p = 1./(1+exp(-(X * (theta+B_i+C_i)))) ; 
    p = min(p,0.9999); %Prevent p from being 1 or 0 due to the precision limits
    p = max(p,0.0001);
    J = (1 / m) * sum(- y .* log(p) .* (1-p).^2 - (1 - y)  .* log(1 - p).* p.^2) + mu_1/2 * norm(theta + C_i - D_i + Q_1i/mu_1, 'fro').^2;
    grad = 1 / m * X' * (-y.*(1-p).^3 + 2.* y .* log(p).*(1-p).^2 .* p + (1-y) .* p.^3 - 2.* (1-y) .* log(1-p) .* (1-p) .* p.^2 ) + mu_1 * (theta + C_i - D_i + Q_1i/mu_1);

   
end

%logistic loss
% function [J,grad] = costFunction_A(theta, X, y, B, C,D,Q_1,mu_1)
%     m = length(y);
%     h = 1./(1+exp(-(X * (theta+B+C))));
%     h = min(h,0.9999); %Prevent p from being 1 or 0 due to the precision limits
%     h = max(h,0.0001);
%     J = (1 / m) * sum(- y .* log(h) - (1 - y) .* log(1 - h)) + mu_1/2 * norm(theta + C - D + Q_1/mu_1, 'fro')^2;
% 
%     disp(J)
%     grad = 1 / m * X' * (h - y) + mu_1 * (theta + C + D + Q_1/mu_1);
% end

function [J,grad] = costFunction_C(theta, X, y,A_i,B_i,D_i,E_i,Q_1i,Q_2i,mu_1,mu_2) %cost function for C
    m = length(y);
    p = 1./(1+exp(-(X * (theta+A_i+B_i)))) ;
    p = min(p,0.9999);
    p = max(p,0.0001);
    J = (1 / m) * sum(- y .* log(p) .* (1-p).^2 - (1 - y)  .* log(1 - p).* p.^2) + mu_1 / 2 * norm(A_i+ theta - D_i + Q_1i/mu_1, 'fro').^2 + mu_2 / 2 * norm(X*theta - E_i + Q_2i / mu_2, 'fro').^2;
    grad = 1 / m * X' * (-y.*(1-p).^3 + 2.* y .* log(p).*(1-p).^2 .* p + (1-y) .* p.^3 - 2.* (1-y) .* log(1-p) .* (1-p) .* p.^2 ) + mu_1 * (A_i + theta - D_i + Q_1i / mu_1) + mu_2 * X' * (X*theta - E_i + Q_2i / mu_2);
end

% soft thresholding: b shoule be the input and t is the learning rate
function [ x ] = soft( b,t )
    x = sign(b).*max(abs(b) - t,0);
end

function [J,grad] = costFunction_B(theta,X,y,A_i,C_i,alpha) 
    m = length(y);
    p = 1./(1+exp(-(X * (A_i+theta+C_i)))) ;
    p = min(p,0.9999);
    p = max(p,0.0001);
    J = (1 / m) * sum(- y .* log(p) .* (1-p).^2 - (1 - y)  .* log(1 - p).* p.^2) + alpha * sum(abs(theta));
    grad = 1 / m * X' * (-y.*(1-p).^3 + 2.* y .* log(p).*(1-p).^2 .* p + (1-y) .* p.^3 - 2.* (1-y) .* log(1-p) .* (1-p) .* p.^2 );
end

function[J] = costFunction_E(eta,X,C,Q_2,mu_2,beta)
    J = beta * sum(sum((abs(eta)))) + mu_2 / 2 * norm(X*C - eta + Q_2/mu_2,'fro').^2;
end

function[J,U_1,S_1,V_1] = costFunction_D(theta,A,C,Q_1,mu_1,gamma)
    [~,S_0,~] = svd(theta);
    trace_norm = sum(sum(S_0));
    [U_1 , S_1, V_1] = svd( A + C + Q_1 / mu_1 );
    J = gamma / mu_1 * trace_norm + 1 / 2 * norm(theta - ( A + C + Q_1 / mu_1 ),'fro').^2; 
end

function[J] = total_loss(X,y,A,B,C,D,E,Q_1,Q_2,mu_1,mu_2, alpha, beta, gamma,c)
    column = 1;
    J = 0;
    [~,S_0,~] = svd(D);
    while(column <= c)
        A_i = A(:,column);
        B_i = B(:,column);
        C_i = C(:,column);
        D_i = D(:,column);
        E_i = E(:,column);
        Q_1i = Q_1(:,column);
        Q_2i = Q_2(:,column);
        y_train_i = y(:,column);
        m = length(y);
        p = 1./(1+exp(-(X * (A_i+B_i+C_i)))) ;
        p = min(p,0.9999);
        p = max(p,0.0001);
%         tl = (1 / m) * sum(- y_train_i .* log(p) .* (1-p).^2 - (1 - y_train_i) .* log(1 - p).* p.^2) + alpha * sum(abs(B_i)) + beta * sum(abs(E_i)) + gamma * sum(sum(S_0)) + mu_1 / 2 * norm(A_i+ C_i - D_i + Q_1i/mu_1, 'fro').^2 - mu_1 / 2 * norm(Q_1i/mu_1,'fro').^2 + mu_2 / 2 * norm(X*C_i - E_i + Q_2i / mu_2, 'fro').^2 - mu_2 / 2 * norm(Q_2i/mu_2,'fro').^2;
         tl = (1 / m) * sum(- y_train_i .* log(p) .* (1-p).^2 - (1 - y_train_i) .* log(1 - p).* p.^2) + mu_1 / 2 * norm(A_i+ C_i - D_i + Q_1i/mu_1, 'fro').^2 - mu_1 / 2 * norm(Q_1i/mu_1,'fro').^2 + mu_2 / 2 * norm(X*C_i - E_i + Q_2i / mu_2, 'fro').^2 - mu_2 / 2 * norm(Q_2i/mu_2,'fro').^2;
        J = J + tl;
        column = column + 1;
    end
    J = J + gamma * sum(sum(S_0))+ alpha * sum(sum(abs(B))) + beta * sum(sum(abs(E))) ;
    
end

% function [ X ] = SVT(M,P,T,delta)
% %Single value thresholding algorithm，SVT
% % function：solve the following optimization problem
% %                  min  ||X||*
% %               s.t. P(X-M) = 0
% % X: recovered matrix
% % M: observed matrix
% % T: single value threshold
% % delta: step size
% % output：X,iterations
%  
% % initialization
% Y = zeros(size(M));
% iterations = 0;
%  
% if nargin < 3
%     T =  sqrt(n1*n2);
% end
% if nargin < 4
%     delta = 1;
% end
% if nargin < 5
%     itermax = 200 ;
% end
% if nargin < 6
%     tol = 1e-7;
% end
%  
% for ii = 1:itermax
%     % singular value threshold operation
%     [U, S, V] = svd(Y, 'econ') ;
%     S = sign(S) .* max(abs(S) - T, 0) ;
%     X = U*S*V' ;
%     % update auxiliary matrix Y
%     Y = Y + delta* P.* (M-X);
%     Y = P.*Y ;
%     % computer error
%     error= norm( P.* (M-X),'fro' )/norm( P.* M,'fro' );
%     if error<tol
%         break;
%     end
%     % update iterations
%     iterations = ii ;
% end
% end



% function [ X ] = SVT(M,T,delta)
% %Single value thresholding algorithm，SVT
% % function：solve the following optimization problem
% %                  min  ||X||*
% %               s.t. P(X-M) = 0
% % X: recovered matrix
% % M: observed matrix
% % T: single value threshold
% % delta: step size
% % output：X,iterations
%  
% % initialization
% Y = zeros(size(M));
% iterations = 0;
%  
% if nargin < 3
%     T =  sqrt(n1*n2);
% end
% if nargin < 4
%     delta = 1;
% end
% if nargin < 5
%     itermax = 200 ;
% end
% if nargin < 6
%     tol = 1e-7;
% end
%  
% for ii = 1:itermax
%     % singular value threshold operation
%     [U, S, V] = svd(Y, 'econ') ;
%     S = sign(S) .* max(abs(S) - T, 0) ;
%     X = U*S*V' ;
%     % update auxiliary matrix Y
%     Y = Y + delta* (M-X);
%     % computer error
%     error= norm( (M-X),'fro' )/norm(M,'fro' );
%     if error<tol
%         break;
%     end
%     % update iterations
%     iterations = ii ;
% end
% end








