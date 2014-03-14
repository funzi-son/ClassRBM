function [W U xB yB hB] = dis_rbm_train(conf,data,label)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training Descriminative RBM                                        %  
% conf: training setting                                             %
% W: weights of connections                                          %
% -*-sontran2012-*-                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data
assert(~isempty(data),'[KRBM-GEN] Data is empty'); 
assert(size(data,1) == size(label,1),'[KRBM-GEN] Number of data and label mismatch'); 
Classes = unique(label)'; %'
lNum = size(Classes,2);
%sm_label = discrete2softmax(label,Classes);
%% initialization
visNum  = size(data,2);
hidNum  = conf.hidNum;
sNum  = conf.sNum;
lr    = conf.params(1);
N     = conf.N;                                                                     % Number of epoch training with lr_1                     
W     = 0.1*randn(visNum,hidNum);
U     = 0.1*randn(lNum,hidNum);
xB    = 0.1*randn(1,visNum);
yB    = 0.1*randn(1,lNum);
hB    = 0.1*randn(1,hidNum);

DW    = zeros(size(W));
DU    = zeros(size(U));
DXB   = zeros(size(xB));
DYB   = zeros(size(yB));
DHB   = zeros(size(hB));
%% evaluation error & early stopping
inc_count = 0;
MAX_INC = conf.MAX_INC;                                                                % If the error increase MAX_INC times continuously, then stop training
%% Average best settings
n_best  = 1;
aW  = size(W);
%% Plotting
h = plot(nan);
%% ==================== Start training =========================== %%
for i=1:conf.eNum
    if i== N+1
        lr = conf.params(2);
    end
    err = Inf('double');
    for j=1:conf.bNum
      X = data((j-1)*sNum+1:j*sNum,:);    
      Y = label((j-1)*sNum+1:j*sNum);
      sm_Y = discrete2softmax(Y,Classes);
      [probs H] = con_probs(W,U,yB,hB,X,lNum);
      if sum(sum(isnan(probs))) ~=0 
          probs
          return;
      end
      % compute the sum 
      logH = logistic(H);     
      A = logH(cumsum([1:sNum:sNum*hidNum;ones(sNum-1,hidNum)]) + repmat(Y*(sNum*hidNum),1,hidNum));    
      B = permute(repmat(probs,[1 1 hidNum]),[1 3 2]);
      B = B.*logH;
      B = sum(B,3);
      C = A - B;
      % update W
       DW = lr*((X'*C)/sNum - conf.params(4)*W) +  conf.params(3)*DW;
       W  = W + DW;       
%        % update U
       DU = lr*((sm_Y'*C)/sNum - conf.params(4)*U) +  conf.params(3)*DU; % Maybe it is wrong here
       U = U + DU;
%        % update bias for X (a in equation)
%         % no need to update
%        % update bias for Y (b in equation)
       DYB = lr*sum(sm_Y - probs,1)/sNum + conf.params(3)*DYB;
       yB = yB + DYB;
%        % update c
       DHB  = lr*sum(C,1)/sNum + conf.params(3)*DHB;
       hB = hB + DHB;
    end
%% Validation
%    sum(sum(W))
%    sum(sum(U))
%    sum(sum(yB))
%    sum(sum(hB))
    
%     err_ = err;
%     [probs H] = con_probs(W,U,yB,hB,data((conf.bNum*sNum + 1):(conf.bNum+conf.vNum)*sNum,:),lNum);      
%     [dummy L] = max(probs,[],2);
%     err = sum((L-1)~=label((conf.bNum*sNum + 1):(conf.bNum+conf.vNum)*sNum,:));
    %% 
%     err_plot(i) = err;
%     axis([0 (conf.eNum+1) 0 1000]);
%     set(h,'YData',err_plot);
%     drawnow;
% %    save(strcat('C:\Pros\Data\XOR\plot_',num2str(conf.params(2),3),'.mat'),'mse_plot');
% %    plot(mse_plot,'XDataSource','real(mse_plot)','YDataSource','imag(mse_plot)')
% %     linkdata on;
    if err > err_
        inc_count = inc_count + 1
    else
        inc_count = 0;
    end
    if inc_count> MAX_INC, break; end;
    fprintf('Epoch %d  : Validation Error = %f\n',i,err);
end

end
