%% Setting for experiment on MNIST dataset
%Dataset
TRN_DAT_FILE = 'mnist_train_dat_10k';
TRN_LAB_FILE = 'mnist_train_lab_10k';

VLD_DAT_FILE = 'mnist_valid_dat_10k';
VLD_LAB_FILE = 'mnist_valid_lab_10k';

TST_DAT_FILE = 'mnist_test_dat_10k';
TST_LAB_FILE = 'mnist_test_lab_10k';

%
TRIAL_NUM  = 1;
R_TYPE     = 'LSP'; % GEN DIS HYB LSP
%Grid search setting
ENUM = 2000;
hidNums = [2000];
lrs    = 0.05;%[0.5 0.7 0.9];
mmts    = [0.02];
csts    = [0.00002];
l_ls    = 1;%[0.01 0.1 0.3 0.5 0.7 1 2];
l_hs    = 0;%[0.01 0.1 0.3 0.5 0.7 1 2];
p_hs    = 0.001;
%g_fs    = 