function output = dis_rbm_classify(model,dat)
% Classify by max energy
% sontran2013
sNum   = size(dat,1);
hidNum = size(model.W,2);
lNum   = size(model.U,1);
%size(repmat(dat*model.W + repmat(model.hidB,sNum,1),[1,1,lNum]))
%size(repmat(reshape((eye(lNum)*model.U)',[1 hidNum lNum]),[sNum,1,1])) 
xxxx = repmat(dat*model.W + repmat(model.hidB,sNum,1),[1,1,lNum]) + ...
    repmat(reshape((eye(lNum)*model.U)',[1 hidNum lNum]),[sNum,1,1]);   
xxxx = reshape(sum(log(1+exp(xxxx)),2),[sNum lNum]);
[~,output] = max(xxxx,[],2);
end