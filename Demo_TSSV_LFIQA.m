% Combining Tensor Slice and Singular Value for Blind Light Field Image
% Quality Assessment(TSSV-LFIQA)
% Code provided by PZY 2020.10.23
clearvars; clearvars -global; clc; close all; warning off;

%% Settings
root = fileparts(mfilename('fullpath')) ;
cd(root) ;
addpath(genpath('./Libraries'));
addpath(genpath('./Functions'));

%% Start Read LF
tic;
dis_path = './LN_dishes_50.bmp';
dis_lf = imread(dis_path);
dis_stack = im2double(permute(reshape(dis_lf,[9, 512, 9, 512, 3]),[2,4,5,1,3])); % 512x512
[h,w,chan,ang,~]=size(dis_stack);

%% Tensor Decompositon
%% (1,2)Get the tensor slice (first slice and other slices)
All_Slice_Hor=cell(1,9);
All_Slice_Ver=cell(1,9);
Fir_Slice_Hor=cell(1,9);
Fir_Slice_Ver=cell(1,9);
Oth_Slice_Hor=cell(1,9);
Oth_Slice_Ver=cell(1,9);
Fir_Slice_Hor_V=cell(1,3);
Fir_Slice_Ver_H=cell(1,3);

temp_Hor_V_c1=[];
temp_Hor_V_c2=[];
temp_Hor_V_c3=[];
temp_Ver_H_c1=[];
temp_Ver_H_c2=[];
temp_Ver_H_c3=[];

% Horizontal
for v=1:9
    All_Slice_Hor{v}=tensor_Slices(dis_stack(:,:,:,v,:));
    Fir_Slice_Hor{v}=All_Slice_Hor{v}(:,:,:,1);
    temp_Hor_V_c1=cat(3,temp_Hor_V_c1,All_Slice_Hor{v}(:,:,1,1));
    temp_Hor_V_c2=cat(3,temp_Hor_V_c2,All_Slice_Hor{v}(:,:,2,1));
    temp_Hor_V_c3=cat(3,temp_Hor_V_c3,All_Slice_Hor{v}(:,:,3,1));
    Oth_Slice_Hor{v}=squeeze(All_Slice_Hor{v}(:,:,1,2:9));
end
temp_Hor_V_c=cat(4,temp_Hor_V_c1,temp_Hor_V_c2,temp_Hor_V_c3);
temphv=zeros(h,w,chan,ang);
for c=1:3
    temphv(:,:,c,:)=tensor_Slices(temp_Hor_V_c(:,:,:,c));
    Fir_Slice_Hor_V{c}=temphv(:,:,c,1);
    if c==1
        Oth_Slice_Hor_V=squeeze(temphv(:,:,c,2:9));
    end
end

% Vertical
for u=1:9
    All_Slice_Ver{u}=tensor_Slices(dis_stack(:,:,:,:,u));
    Fir_Slice_Ver{u}=All_Slice_Ver{u}(:,:,:,1);
    temp_Ver_H_c1=cat(3,temp_Ver_H_c1,All_Slice_Ver{u}(:,:,1,1));
    temp_Ver_H_c2=cat(3,temp_Ver_H_c2,All_Slice_Ver{u}(:,:,2,1));
    temp_Ver_H_c3=cat(3,temp_Ver_H_c3,All_Slice_Ver{u}(:,:,3,1));
    Oth_Slice_Ver{u}=squeeze(All_Slice_Ver{u}(:,:,1,2:9));
end
temp_Ver_H_c=cat(4,temp_Ver_H_c1,temp_Ver_H_c2,temp_Ver_H_c3);
tempvh=zeros(h,w,chan,ang);
for c=1:3
    tempvh(:,:,c,:)=tensor_Slices(temp_Ver_H_c(:,:,:,c));
    Fir_Slice_Ver_H{c}=tempvh(:,:,c,1);
    if c==1
        Oth_Slice_Ver_H=squeeze(tempvh(:,:,c,2:9));
    end
end


%% (3)Get the singular values of intra-stacked SAIs
V_Intra_Hor=cell(1,9);
V_Intra_Ver=cell(1,9);
for vv=1:9
    V_Intra_Hor{vv}=tensor_SAI_SVD_value(dis_stack(:,:,:,vv,:));
end
for uu=1:9
    V_Intra_Ver{uu}=tensor_SAI_SVD_value(dis_stack(:,:,:,:,uu));
end

%% (4)Get the singular values of inter-stacked SAIs
V_mat=tensor_SAI_SVD_value(dis_stack);

%% Feature Extraction
%% 1. Tensor Slice Spatial Feature (TSSF) including
% the first slice sharpness measurement and the other slices information distribution

%%  (1)the first slice sharpness measurement
data_eachf = zeros(2,9,3);
data_fea = zeros(2,3);

Grad_Var_each_h= data_eachf;
Grad_Var_each_v= data_eachf;

Grad_Var_ave_h= data_fea ;
Grad_Var_ave_v= data_fea ;
Grad_Var_ave_hv=data_fea ;

Grad_Var_each_h_sum= data_fea ;
Grad_Var_each_v_sum= data_fea ;
Grad_Var_each_hv_sum= data_fea ;

% Horizontal
for hor=1:9
    for c=1:3
        temp= map_pzy(abs(Fir_Slice_Hor{hor}(:,:,c)));
        [GM1_h,RO1_h,~]=LF_Gradient(temp*255);
        Grad_Var_each_h(1,hor,c)=VarInformation(GM1_h, 2);
        Grad_Var_each_h(2,hor,c)=VarInformation(RO1_h, 1);
    end
end
Grad_Var_ave_h=squeeze(mean(Grad_Var_each_h,2));

for c=1:3
    temp_Hor_V= map_pzy(abs(Fir_Slice_Hor_V{c}));
    [GM1_h_sum,RO1_h_sum,~]=LF_Gradient(temp_Hor_V*255);
    Grad_Var_each_h_sum(1,c)=VarInformation(GM1_h_sum, 2);% compute the statistics variance of GM
    Grad_Var_each_h_sum(2,c)=VarInformation(RO1_h_sum, 1);% compute the statistics variance of RO
end


% Vertical
for ver=1:9
    for c=1:3
        temp= map_pzy(abs(Fir_Slice_Ver{ver}(:,:,c)));
        [GM1_v,RO1_v,~]=LF_Gradient(temp*255);
        Grad_Var_each_v(1,ver,c)=VarInformation(GM1_v, 2);% compute the statistics variance of GM
        Grad_Var_each_v(2,ver,c)=VarInformation(RO1_v, 1);% compute the statistics variance of RO
    end
end
Grad_Var_ave_v=squeeze(mean(Grad_Var_each_v,2));

temp_hv_each=cat(3,Grad_Var_ave_h,Grad_Var_ave_v);
Grad_Var_ave_hv=squeeze(mean(temp_hv_each,3));

for c=1:3
    temp_Ver_H= map_pzy(abs(Fir_Slice_Ver_H{c}));
    [GM1_v_sum,RO1_v_sum,~]=LF_Gradient(temp_Ver_H*255);
    Grad_Var_each_v_sum(1,c)=VarInformation(GM1_v_sum, 2);% compute the statistics variance of GM
    Grad_Var_each_v_sum(2,c)=VarInformation(RO1_v_sum, 1);% compute the statistics variance of RO
end

temp_hv_each_sum=cat(3,Grad_Var_each_h_sum,Grad_Var_each_v_sum);
Grad_Var_each_hv_sum=squeeze(mean(temp_hv_each_sum,3));

Feature_slice_C1=cat(1,Grad_Var_ave_hv(:,1),Grad_Var_each_hv_sum(:,1));
Feature_slice_C2=cat(1,Grad_Var_ave_hv(:,2),Grad_Var_each_hv_sum(:,2));
Feature_slice_C3=cat(1,Grad_Var_ave_hv(:,3),Grad_Var_each_hv_sum(:,3));
Feature_QX1= cat(1,Feature_slice_C1,Feature_slice_C2,Feature_slice_C3);

sprintf('(1)the first slice sharpness measurement is done ')
%% (2)the other slices information distribution


data_Each_hor_pp = zeros(2,8,9);
data_Each_ver_pp = zeros(2,8,9);
data_Interv=zeros(2,8);


Slice_h_Each_pp=data_Interv;
Slice_v_Each_pp=data_Interv;
Slice_hv_Each_pp=data_Interv;

new_Slice_h_Each_pp=data_Interv;
new_Slice_v_Each_pp=data_Interv;
new_Slice_hv_Each_pp=data_Interv;

for hor=1:9
    for sli=2:9 
        temp= map_pzy(abs(Oth_Slice_Hor{hor}(:,:,sli-1)));
        data_Each_hor_pp(:,sli-1,hor)=LF_Entropy(temp,2);
    end
end

for ver=1:9
    for sli=2:9 
        temp= map_pzy(abs(Oth_Slice_Ver{ver}(:,:,sli-1)));
        data_Each_ver_pp(:,sli-1,ver)=LF_Entropy(temp,2);
    end
end
Slice_h_Each_pp=squeeze(mean(data_Each_hor_pp,3));
Slice_v_Each_pp=squeeze(mean(data_Each_ver_pp,3));
temp_hv=cat(3,Slice_h_Each_pp,Slice_v_Each_pp);
Slice_hv_Each_pp=squeeze(mean(temp_hv,3));

for numi=2:9
    temp_a= map_pzy(abs(Oth_Slice_Hor_V(:,:,numi-1)));
    new_Slice_h_Each_pp(:,numi-1)=LF_Entropy(temp_a,2);
    temp_b= map_pzy(abs(Oth_Slice_Ver_H(:,:,numi-1)));
    new_Slice_v_Each_pp(:,numi-1)=LF_Entropy(temp_b,2);
end
temp_hv_new=cat(3,new_Slice_h_Each_pp,new_Slice_v_Each_pp);
new_Slice_hv_Each_pp=squeeze(mean(temp_hv_new,3));

Feature_Entropy2= cat(1,Slice_hv_Each_pp(:),new_Slice_hv_Each_pp(:));

sprintf('(2)the other slices information distribution is done ')
%% 2. Singular Value Angular Feature (SVAF) including
% singular values of intra-stacked SAIs and inter-stacked SAIs

%% (3)the percentage of the singular values of intra-stacked SAIs

data_c = zeros(9,1);
singular_h_intra=data_c;
singular_v_intra=data_c;

V_Intra_H=[];
for Hor=1:9
    V_Intra_Hor_mat=cell2mat(V_Intra_Hor{Hor});
    V_Intra_Hor_mat=V_Intra_Hor_mat/sum(V_Intra_Hor_mat);
    V_Intra_H=[V_Intra_H,V_Intra_Hor_mat];
end

V_Intra_V=[];
for Ver=1:9
    V_Intra_Ver_mat=cell2mat(V_Intra_Ver{Ver});
    V_Intra_Ver_mat=V_Intra_Ver_mat/sum(V_Intra_Ver_mat);
    V_Intra_V=[V_Intra_V,V_Intra_Ver_mat];
end
singular_h_intra(:,1)=mean(V_Intra_H,2);
singular_v_intra(:,1)=mean(V_Intra_V,2);


ratio_intra=0.5;
num_Singular_intra=ceil(size(singular_h_intra,1)*ratio_intra);
if num_Singular_intra*2>size(singular_h_intra,1)
    num_Singular_intra_head=num_Singular_intra-1;
    num_Singular_intra_tail=num_Singular_intra;
else
    num_Singular_intra_head=num_Singular_intra;
    num_Singular_intra_tail=num_Singular_intra;
end

Feature_h_intra=cat(1,singular_h_intra(1:num_Singular_intra_head,:),singular_h_intra(end-num_Singular_intra_tail+1:end,:));
Feature_v_intra=cat(1,singular_v_intra(1:num_Singular_intra_head,:),singular_v_intra(end-num_Singular_intra_tail+1:end,:));
Feature_svd_intra3=cat(1,Feature_h_intra,Feature_v_intra);

sprintf('(3)the percentage of the singular values of intra-stacked SAIs is done ')
%% (4)the percentage of the singular values of inter-stacked SAIs
singular_h_inter=data_c;
singular_v_inter=data_c;

singular_h_inter(:,1)=V_mat{2}/sum(V_mat{2});
singular_v_inter(:,1)=V_mat{1}/sum(V_mat{1});

ratio_inter=0.5;
num_Singular_inter=ceil(size(singular_h_inter,1)*ratio_inter);
if num_Singular_inter*2>size(singular_h_inter,1)
    num_Singular_inter_head=num_Singular_inter-1;
    num_Singular_inter_tail=num_Singular_inter;
else
    num_Singular_inter_head=num_Singular_inter;
    num_Singular_inter_tail=num_Singular_inter;
end

Feature_h_inter=cat(1,singular_h_inter(1:num_Singular_inter_head,:),singular_h_inter(end-num_Singular_inter_tail+1:end,:));
Feature_v_inter=cat(1,singular_v_inter(1:num_Singular_inter_head,:),singular_v_inter(end-num_Singular_inter_tail+1:end,:));
Feature_svd_inter4=cat(1,Feature_h_inter,Feature_v_inter);

sprintf('(4)the percentage of the singular values of inter-stacked SAIs is done ')

Feature_all = cat(1,Feature_QX1,Feature_Entropy2,Feature_svd_intra3,...
    Feature_svd_inter4);
toc;

%% Quality Prediction
load('model.mat')
len=size(Feature_all,1);
for i=1:len
    [Feature_all(i,:)] = normalization_ML(Feature_all(i,:),0,1,max_m(i),min_m(i));
end
Pred_quality = svmpredict(1.39, Feature_all', model);
sprintf('Quality Score Prediction:  %.2f ',Pred_quality)


