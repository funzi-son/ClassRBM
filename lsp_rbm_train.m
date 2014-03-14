function model = lsp_rbm_train(conf)
% Label-sparsity constrained RBM
% sontran2013

global trn_dat trn_lab vld_dat vld_lab tst_dat tst_lab;


%prepare
[SZ,visNum] = size(trn_dat);
hidNum = conf.hidNum;
sNum   = conf.sNum;
labNum = size(unique(trn_lab),1);
if conf.bNum == 0, bNum   = round(SZ/sNum); end

mmm = max(visNum,hidNum);
model.W    = (1/sqrt(mmm))*(2*rand(visNum,hidNum)-1);
mmm = max(labNum,hidNum);
model.U    = (1/sqrt(mmm))*(2*rand(labNum,hidNum)-1);

model.visB = 0.00*randn(1,visNum);
model.hidB = 0.00*randn(1,hidNum);
model.labB = 0.00*randn(1,labNum);

WD  = zeros(size(model.W));
UD  = zeros(size(model.U));
visBD = zeros(size(model.visB));
hidBD = zeros(size(model.hidB));
labBD = zeros(size(model.labB));


lr = conf.params(1);
tic;
vld_best = 0;
vld_prv  = 0;
es_count = 0;
e = 0;
while es_count<=conf.E_STOP && e<1000
    
    e = e+1;
    inx = randperm(SZ);
    %dat = dat(inx,:);
    %lab = lab(inx);
    
    res_e = 0;
    trn_acc = 0;
    hspr  = 0;
    lspr  = 0;
    
    if e==conf.N+1, lr = conf.params(2); end 
    for b=1:bNum
        w_diff = 0; u_diff = 0; v_diff = 0; h_diff = 0; l_diff = 0;
        iiii = inx((b-1)*sNum+1:b*sNum);
        visP = trn_dat(iiii,:);
        labP = trn_lab(iiii) + 1;
        
       hidP = logistic(visP*model.W + model.U(labP,:) + repmat(model.hidB,sNum,1));
%         hidP = logistic(visP*model.W + repmat(model.hidB,sNum,1));
        hidPs = hidP>rand(size(hidP));
        hidNs = hidPs;
        if conf.gen_f>0
            %% gibb sampling
            for g=1:conf.gNum
                visN = logistic(bsxfun(@plus,hidNs*model.W',model.visB));
                visNs = visN>rand(size(visN));
                labN = softmax(exp(bsxfun(@plus,hidNs*model.U',model.labB)));

                hidN = logistic(visNs*model.W + model.U(labN,:) + repmat(model.hidB,sNum,1));            
    %             hidN = logistic(visNs*model.W + repmat(model.hidB,sNum,1));
                hidNs = hidN>rand(size(hidN));
            end        
            res_e = res_e + sum(sqrt(sum((visP - visNs).^2,2))/visNum,1)/sNum;
            %
    %        acc_e = acc_e + sum(sum(labP == labN))/sNum;
            %% updating from log-likelihood        
            w_diff = conf.gen_f*(visP'*hidP - visNs'*hidN)/sNum;        

            s_labP = discrete2softmax(labP,labNum);
            s_labN = discrete2softmax(labN,labNum);

            u_diff = conf.gen_f*(s_labP'*hidP - s_labN'*hidN)/sNum;                                
            v_diff = conf.gen_f*sum(visP - visNs,1)/sNum;               
            h_diff = conf.gen_f*sum(hidPs - hidNs,1)/sNum;                
            l_diff = sum(s_labP - s_labN,1)/sNum;
        
        end
        %% Label Sparsity constrains
        if conf.lambda_l >0            
            [cprobs H] = con_probs(model.W,model.U,model.labB,model.hidB,visP);           
           
            inv_labprobs = (1-cprobs([1:sNum]' + (labP-1)*sNum));
            inv_labprobs_norm = inv_labprobs.^(conf.norm-1);
            
            H = logistic(H);
            hk_x = H.*repmat(reshape(cprobs,[sNum 1 labNum]),[1 hidNum 1]);
            e_hh = sum(hk_x,3);
            
            hhh = bsxfun(@times,hidP - e_hh,inv_labprobs_norm);
            
            w_diff = w_diff + conf.lambda_l*(visP'*hhh)/sNum;            
            
            s_Y = discrete2softmax(labP,labNum);
            hy_x = H.*repmat(reshape(s_Y,[sNum 1 labNum]),[1 hidNum 1]);
            
            u_diff = u_diff + conf.lambda_l*reshape(sum((hy_x-hk_x).*repmat(inv_labprobs_norm,[1 hidNum labNum]),1),[hidNum labNum])'/sNum;                        
            h_diff = h_diff + conf.lambda_l*sum(hhh,1)/sNum;            
            l_diff = l_diff + conf.lambda_l*sum(bsxfun(@times,s_Y - cprobs,inv_labprobs_norm),1)/sNum;            
            lspr = lspr + sum(inv_labprobs.^2,1)/sNum;
        end
        %% Hidden Sparsity contrains
        if conf.lambda_h >0                              
           hidI = (visP*model.W +  repmat(mdel.hidB,sNum,1));           
           pppp = (conf.p - sum(hidP,1)/sNum);
           
           w_diff = w_diff + conf.lambda_h*(repmat(pppp,visNum,1).*(visP'*((hidP.^2).*exp(-hidI))));
           %u_diff = 
           h_diff = h_diff + conf.lambda*(pppp.*(sum((hidp.^2).*exp(-hidI),1)/sNum));                                 
           %l_diff
           hspr = hspr + sum((conf.p - mean(hidP)).^2,2);
        end
       
        % Updating
        WD = lr*(w_diff - conf.params(4)*model.W) + conf.params(3)*WD;
        model.W = model.W + WD;
        UD = lr*(u_diff - conf.params(4)*model.U) + conf.params(3)*UD;
        model.U = model.U + UD;
        visBD = lr*v_diff + conf.params(3)*visBD;
        model.visB = model.visB + visBD;
        hidBD = lr*h_diff + conf.params(3)*hidBD;
        model.hidB = model.hidB + hidBD;
        labBD = lr*l_diff + conf.params(3)*labBD;
        model.labB = model.labB + labBD;
        % Get training accuracy
        trn_acc = trn_acc + sum(labP == rbm_classify(model,visP,2))/sNum;
    end
    %output = rbm_classify(model,tst_dat,1); % classify using max-reconstruction probs
    %acc_r  = sum(sum(output==tst_lab+1))/size(tst_lab,1);
    
    output = rbm_classify(model,vld_dat,2); % classify using max-energy (cond probs) for validation set
    vld_acc  = sum(sum(output==vld_lab+1))/size(vld_lab,1);
    
    output = rbm_classify(model,tst_dat,2); % classify using max-energy (cond  probs) for test set
    tst_acc  = sum(sum(output==tst_lab+1))/size(tst_lab,1);
        
    
    hspr = hspr/bNum;
    lspr = lspr/bNum;
    trn_acc= trn_acc/bNum; % training accuracy (aprx)
    
    fprintf('[Epoch %.4d] res_e = %.5f || hspr = %.5f || lspr = %.5f ||trn_acc = %.5f ||vld_acc = %.5f || acc_p = %.5f\n',...
        e,res_e, hspr,lspr,trn_acc,vld_acc,tst_acc);        
    
    if ~isempty(conf.vis_dir)
        save_images(visN,100,28,28,strcat(conf.vis_dir,num2str(e),'.bmp'));
    end
    
    % Early stopping
    if vld_acc<vld_prv        
        es_count = es_count + 1;         
    else
        es_count = 0;        
    end
    vld_prv = vld_acc;
end
toc;
end