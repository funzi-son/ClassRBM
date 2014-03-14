function [probs H] = con_probs(W,U,labB,hidB,X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Calculate P(y|x) in RBM                                                %
% W: weights matrix of X and hiddens, U: weights matrix of Y and hiddens %
% labB: bias on Y, hidB: bias on hidden (bias on visible is not needed   %
% X: matrix mxn contains m input vectors x                               %
% class_size: number of labels                                           %
% Return: matrix of m x class_size : P(label|x)                          %
% -*-sontran2013-*-                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hNum = size(W,2);
assert(hNum == size(U,2) && hNum == size(hidB,2),'[CON-PROBS] Matrices dimension are not consistent');
[m xNum] = size(X);
assert(xNum == size(W,1),'[CON-PROBS] Input size does not match its weight matrix');
yNum = size(U,1);
%% Initializing the matrix H(m,class_size,hNum) each stores:
% \sum_i x_iw_{ij} + u_{yj} + c_j
H = repmat(X*W,[1 1 yNum]);
for k=1:yNum
    H(:,:,k) = H(:,:,k) + repmat(U(k,:),m,1) + repmat(hidB,m,1);
%     probs(:,k) = exp(b(k))*prod(1 + exp(H(:,:,k)),2);
    log_probs(:,k) = labB(k)+ sum(log(1 + exp(H(:,:,k))),2);  
end
% normalizing probs in log-space and convert back to decimal-place
probs = exp(log_probs - repmat(max(log_probs,[],2),1,yNum));
%% Compute conditional prob for all possible layers
probs = probs./repmat(sum(probs,2),1,yNum);
end