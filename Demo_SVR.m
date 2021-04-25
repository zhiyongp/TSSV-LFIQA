%2020.6.26
%(1)Tensor Slice Spatial Feature (TSSF) including
%the first slice(three color channels) sharpness measurement 
%and the other slices information distribution

%(2)Singular Value Angular Feature (SVAF) including
% singular values of intra-stacked SAIs and inter-stacked SAIs

%% Before performing SVR, it is important to scale the features of the entire feature set, 
% generally to [-1,1] or [0,1]
close all;clear all;clc;    warning off

%% Navigate to the directory where the current M file is located
root = fileparts(mfilename('fullpath')) ;
cd(root) ;
addpath(genpath('./Libraries'));
addpath(genpath('Functions'));

%% Import MOS 
tic
load('./win5mos.mat');  % The MOS of all images in the Win5-LID database, sorted by scene
data_dmos=win5mos;

%% Tensor Slice Spatial Feature (TSSF)
% % the first slice(three color channels) sharpness measurement £¨by Gradient£©--------------------------------------------------------------

% First Channel
load('./Feature/First_Grad_Feature.mat');
%%% The first slice in each row (column)
Feature_slice1= Grad_Var_ave_hv(1:2,:);
%%% The first slice between each row (column)
Feature_slice2= Grad_Var_each_hv_sum(1:2,:);
Feature_slice=cat(1,Feature_slice1,Feature_slice2);

% Second Channel
load('./Feature/First_Grad_Feature_Color.mat'); 
%%% The first slice in each row (column)
Feature_slice1_c1= Grad_Var_ave_hv_c1(1:2,:);
%%% The first slice between each row (column)
Feature_slice2_c1= Grad_Var_each_hv_sum_c1(1:2,:);
Feature_slice_c1=cat(1,Feature_slice1_c1,Feature_slice2_c1);

% Third Channel
%%% The first slice in each row (column)
Feature_slice1_c2= Grad_Var_ave_hv_c2(1:2,:);
%%% The first slice between each row (column)
Feature_slice2_c2= Grad_Var_each_hv_sum_c2(1:2,:);
Feature_slice_c2=cat(1,Feature_slice1_c2,Feature_slice2_c2);

Feature_QX1= cat(1,Feature_slice,Feature_slice_c1,Feature_slice_c2);

% % the other slices information distribution£¨by Entropy£©--------------------------------------------------------------
load('./Feature/Other_Entropy_Feature.mat'); %
temp_slice_each=reshape(Slice_hv_Each_pp(1:2,1:8,:),[2*8,length(data_dmos)]);
temp_slice_new=reshape(new_Slice_hv_Each_pp(1:2,1:8,:),[2*8,length(data_dmos)]);
Feature_Entropy2= cat(1,temp_slice_each,temp_slice_new);

%% Singular Value Angular Feature (SVAF)

% % inter-stacked SAIs£¨by the percentage of singular values £©--------------------------------------------------------------
load('./Feature/SinValue_inter_Feature.mat');

ratio_inter=0.5;
num_Singular_inter=ceil(size(singular_h_inter,1)*ratio_inter);
if num_Singular_inter*2>size(singular_h_inter,1)
    num_Singular_inter_head=num_Singular_inter-1;
    num_Singular_inter_tail=num_Singular_inter;
else
    num_Singular_inter_head=num_Singular_inter;
    num_Singular_inter_tail=num_Singular_inter;
end
%inter-row
Feature1_h_inter=cat(1,singular_h_inter(1:num_Singular_inter_head,:),singular_h_inter(end-num_Singular_inter_tail+1:end,:));
%inter-column
Feature1_v_inter=cat(1,singular_v_inter(1:num_Singular_inter_head,:),singular_v_inter(end-num_Singular_inter_tail+1:end,:));

%inter-row and inter-column
Feature_svd_inter3=cat(1,Feature1_h_inter,Feature1_v_inter);

% % intra-stacked SAIs£¨by the percentage of singular values £©--------------------------------------------------------------
load('./Feature/SinValue_intra_Feature.mat');
ratio_intra=0.5;
num_Singular_intra=ceil(size(singular_h_intra,1)*ratio_intra);
if num_Singular_intra*2>size(singular_h_intra,1)
    num_Singular_intra_head=num_Singular_intra-1;
    num_Singular_intra_tail=num_Singular_intra;
else
    num_Singular_intra_head=num_Singular_intra;
    num_Singular_intra_tail=num_Singular_intra;
end
%intra-row
Feature1_h_intra=cat(1,singular_h_intra(1:num_Singular_intra_head,:),singular_h_intra(end-num_Singular_intra_tail+1:end,:));
%intra-column
Feature1_v_intra=cat(1,singular_v_intra(1:num_Singular_intra_head,:),singular_v_intra(end-num_Singular_intra_tail+1:end,:));
%intra-row and intra-column
Feature_svd_intra4=cat(1,Feature1_h_intra,Feature1_v_intra);

%% Feature test
Feature_test = cat(1,Feature_QX1,Feature_Entropy2,Feature_svd_inter3,...
    Feature_svd_intra4);
len=size(Feature_test,1);
for i=1:len 
    [Feature_test(i,:),max_m(i),min_m(i)] = normalization_ML(Feature_test(i,:),0,1);
end

%% Remove the reference image
index_temp = find(data_dmos);  
dmos_use = data_dmos(index_temp);
feature_use = Feature_test(:,index_temp);
p=cell(1,100);
q=cell(1,100);
ms=zeros(1,100);

%% Cost and gamma (Two hyperparameters C, g in the RBF kernel in SVR)
for  cishu=1:100
num_length = length(index_temp);
index_all = randperm(num_length);
index_train = index_all( 1 : round(num_length*0.8));
index_test = index_all( round(num_length*0.8)+1 : end); 
dmos_train = dmos_use( index_train );
dmos_test  = dmos_use( index_test );
feature_train = feature_use(:,index_train );
feature_test  = feature_use(:,index_test );

bestp = 0.1;   %%% The default is 0.1;
% In the './Libraries/SVR/Guide.pdf', C is taken  (-5:2:15), g is taken (3:-2:-15)
% Below C is taken (-4:1:15), g is taken (5:-1:-4)

    for c=1:20
        for g=1:10
            cost = 2^(c-5);
            gamma = 2^(g-5);
            c_str = sprintf('%f',cost);
            g_str = sprintf('%.2f',gamma);
            libsvm_options = ['-s 3 -t 2 -g ',num2str(gamma),' -c ',num2str(cost),'-p ',num2str(bestp)];
            model = svmtrain(dmos_train,feature_train',libsvm_options); %#ok<*SVMTRAIN>
            [predict_score, ~, ~] = svmpredict(zeros(size(dmos_test)), feature_test', model);
            [pearson_cc_1,spearman_srocc1(c,g), rmse_1] = svr_IQA(predict_score, dmos_test);
        end
    end
    ms(cishu) = max(spearman_srocc1(:));
    [p{cishu},q{cishu}] =find(spearman_srocc1 == ms(cishu));
end
[p,q]=find_index_new(p,q);
% p=10;
% q=1;

cost_better = 2^(p-5);
gamma_better = 2^(q-5);
libsvm_options = ['-s 3 -t 2 -g ',num2str(gamma_better),' -c ',num2str(cost_better),'-p ',num2str(bestp)];
%-s refers to SVM_type, the default is 0, and 3 is epsilon_SVR; 
%-t refers to kernel_type, the default is 2, which is RBF kernel.
%-g -c are the hyperparameters in the RBF kernel; -p refers to epsilon, and the default is 0.1.

%% Training-testing split is performed 1000 times
for i = 1:1000
    i 
    index_all = randperm(num_length); 
    index_train = index_all( 1 : round(num_length*0.8)); 
    index_test = index_all( round(num_length*0.8)+1 : end); 
    
    dmos_train = dmos_use( index_train );
    dmos_test  = dmos_use( index_test );
    feature_train = feature_use(:,index_train );
    feature_test  = feature_use(:,index_test );
    
    %% SVR train and test
    svm_model = svmtrain(dmos_train , feature_train' , libsvm_options);  % svr train
    [pred, ~ , ~ ] = svmpredict(zeros(size(dmos_test)), feature_test', svm_model); % svr test
    
    %% Logistic mapping
    [ pearson_cc(1,i), spearman_srocc(1,i),rmse(1,i)] = svr_IQA(pred, dmos_test);
end

%% Performance
plcc_mean = mean(pearson_cc);srocc_mean = mean(spearman_srocc);rmse_mean = mean(rmse);
data_mean = [plcc_mean, srocc_mean,rmse_mean];  % mean
plcc_median = median(pearson_cc);srocc_median = median(spearman_srocc);rmse_median = median(rmse);
data_median = [plcc_median, srocc_median,rmse_median]; % median

toc
