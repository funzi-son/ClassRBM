function L = softmax(inp)
% Softmax function
% sontran2013
[m n] = size(inp);
prob = inp./repmat(sum(inp,2),1,n);
p = cumsum(prob,2)>repmat(rand(m,1),1,n);
L = n + 1 - sum(p,2);
end
