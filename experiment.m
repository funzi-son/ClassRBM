function experiment(exp_setting)
%Experiment for classification RBM
%sontran2013
eval(exp_setting);

fprintf('...reach here ...\n');
global trn_dat trn_lab vld_dat vld_lab tst_dat tst_lab;

vars = whos('-file', TRN_DAT_FILE);
trn_dat = load(TRN_DAT_FILE,vars(1).name);
trn_dat = trn_dat.(vars(1).name);    

vars = whos('-file', TRN_LAB_FILE);
trn_lab = load(TRN_LAB_FILE,vars(1).name);
trn_lab = trn_lab.(vars(1).name);    

vars = whos('-file', VLD_DAT_FILE);
vld_dat = load(VLD_DAT_FILE,vars(1).name);
vld_dat = vld_dat.(vars(1).name);    

vars = whos('-file', VLD_LAB_FILE);
vld_lab = load(VLD_LAB_FILE,vars(1).name);
vld_lab = vld_lab.(vars(1).name);    

vars = whos('-file', TST_DAT_FILE);
tst_dat = load(TST_DAT_FILE,vars(1).name);
tst_dat = tst_dat.(vars(1).name);    

vars = whos('-file', TST_LAB_FILE);
tst_lab = load(TST_LAB_FILE,vars(1).name);
tst_lab = tst_lab.(vars(1).name);
    
    
        % Common params
        conf.hidNum = 500;
        conf.eNum   = 20;        
        conf.sNum   = 100;
        conf.bNum   = 0; % calculate in function
        conf.gNum   = 1;
        conf.params = [0.2 0.2 0.02 0.00002];
        
        % Generative params
        conf.alpha = 1; % Not used 
        
        % Label sparsity
        conf.lambda_l = 1;
        % Clasification Norm
        conf.norm     = 1;
        %generative factor
        conf.gen_f = 1-conf.lambda_l;
        % Hidden sparsity
        conf.lambda_h = 0;
        conf.p        = 0;
        
        %Early stopping
        conf.E_STOP  = 15;
        
        conf.N       = 10;
        conf.plot_   = 0;
        conf.vis_dir = [];
                
        
        if strcmp(R_TYPE,'GEN') % Generative RBM
            model  = gen_rbm_train(conf);
%             acc = gen_rbm_classify(model,TST_DAT_FILE,TST_LAB_FILE);
        elseif strcmp(R_TYPE,'DIS') % Discriminative RBM
            model  = dis_rbm_train(conf);
%             acc = dis_rbm_classify(model,TST_DAT_FILE,TST_LAB_FILE);
        elseif strcmp(R_TYPE,'HYB') % Hybrid RBM
            model  = hyb_rbm_train(conf);
%             acc = hyb_rbm_classify(model,TST_DAT_FILE,TST_LAB_FILE);
        elseif strcmp(R_TYPE,'LSP') % Label Sparsity RBM
            model  = lsp_rbm_train(conf);
%             acc = lsp_rbm_classify(model,TST_DAT_FILE,TST_LAB_FILE);
        else
            fprintf('No learning model is set for this experiment!!!!\n');
            return;
        end                    
  

end

