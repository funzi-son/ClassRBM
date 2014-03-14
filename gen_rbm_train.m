function model = gen_rbm_train(conf,trn_dat_file,trn_lab_file,tst_dat_file,tst_lab_file)
% Train generative RBM with label
% sontran2013
%% load file
vars = whos('-file', trn_dat_file);
A = load(trn_dat_file,vars(1).name);
dat = A.(vars(1).name);
vars = whos('-file', trn_lab_file);
A = load(trn_lab_file,vars(1).name);
lab = A.(vars(1).name);

if ~isempty(tst_dat_file) && ~isempty(tst_lab_file)
    vars = whos('-file', tst_dat_file);
    A = load(tst_dat_file,vars(1).name);
    tdat = A.(vars(1).name);
    vars = whos('-file', tst_lab_file);
    A = load(tst_lab_file,vars(1).name);
    tlab = A.(vars(1).name);
end
clear A;
%% setting up
hidNum = conf.hidNum;
visNum = size(dat,2);
labNum = size(unique(lab),1);
sNum = conf.sNum;

model.W = 0.1*randn(visNum,hidNum);
model.U = 0.1*randn(labNum,hidNum);

model.visB = zeros(1,visNum);
model.hidB = zeros(1,hidNum);
model.labB = zeros(1,labNum);

WD    = zeros(size(model.W));
UD    = zeros(size(model.U));
visBD = zeros(size(model.visB));
hidBD = zeros(size(model.hidB));
labBD = zeros(size(model.labB));

%% running
lr = conf.params(1);
for e=1:conf.eNum;
    inx = randperm(size(dat,1));    
    
    res_e = 0;
    acc_e = 0;
    spr_e = 0;
    
    if e==conf.N+1, lr = conf.params(2); end 
    for b=1:conf.bNum
        iiii = inx((b-1)*sNum+1:b*sNum);
        visP = dat(iiii,:);
        labP = lab(iiii) + 1;
        
       hidP = logistic(visP*model.W + model.U(labP,:) + repmat(model.hidB,sNum,1));
%         hidP = logistic(visP*model.W + repmat(model.hidB,sNum,1));
        hidPs = hidP>rand(size(hidP));
        hidNs = hidPs;
        %% gibb sampling
        for g=1:conf.gNum
            visN = logistic(bsxfun(@plus,hidNs*model.W',model.visB));
            visNs = visN>rand(size(visN));
            labN = softmax(exp(bsxfun(@plus,hidNs*model.U',model.labB)));
            
            hidN = logistic(visNs*model.W + model.U(labN,:) + repmat(model.hidB,sNum,1));            
%             hidN = logistic(visNs*model.W + repmat(model.hidB,sNum,1));
            hidNs = hidN>rand(size(hidN));
        end        
        res_e = res_e + sum(sqrt(sum((visP - visNs).^2,2)/visNum),1)/sNum;
        if ~isempty(tdat)
            acc_e = acc_e + sum(tlab+1 == gen_rbm_classify(model,tdat+1,labNum,conf.clss_opt))/tNum;            
        end
        %% updating
        w_diff = (visP'*hidP - visNs'*hidN)/sNum;
        WD = lr*(w_diff - conf.params(4)*model.W) + conf.params(3)*WD;
        model.W = model.W + WD;
        
        s_labP = discrete2softmax(labP,labNum);
        s_labN = discrete2softmax(labN,labNum);
        u_diff = (s_labP'*hidP - s_labN'*hidN)/sNum;        
        UD = lr*(u_diff - conf.params(4)*model.U) + conf.params(3)*UD;
        model.U = model.U + UD;
        
        visBD = lr*sum(visP - visNs,1)/sNum + conf.params(3)*visBD;
        model.visB  = model.visB + visBD;
        
        hidBD = lr*sum(hidPs - hidNs,1)/sNum + conf.params(3)*hidBD;
        model.hidB  = model.hidB + hidBD;
        
        labBD = lr*sum(s_labP - s_labN,1)/sNum + conf.params(3)*labBD;
        model.labB  = model.labB + labBD;
        
        %% Sparsity contrains
        if conf.lambda >0
           hidI = (visP*model.W +  repmat(model.hidB,sNum,1));
           hidN = logistic(hidI);
           pppp = (conf.p - sum(hidN,1)/sNum);
           %model.W    = model.W   + lr*conf.lambda*(repmat(pppp,visNum,1).*(visP'*((hidN.^2).*exp(-hidI))));
           model.hidB = model.hidB + lr*conf.lambda*(pppp.*(sum((hidN.^2).*exp(-hidI),1)/sNum));           
           spr_e = spr_e + sum((conf.p - mean(hidN)).^2,2);
        end
    end
    fprintf('[Epoch %.3d] res_e = %.5f ||clf_e = %.5f || spr_e = %.3f \n',e,res_e/conf.bNum,acc_e/conf.bNum,spr_e/conf.bNum);
    if ~isempty(conf.vis_dir)
        save_images(visN,100,28,28,strcat(conf.vis_dir,num2str(e),'.bmp'));
    end
end   
end