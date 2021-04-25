function V_mat=tensor_SAI_SVD_value(im)
% Function of Tensor decomposition
% S = HxWxCxVxU
S=tensor(im);
if size(im,4)==1
    D=5;%%Channels that need dimensionality reduction
end
if size(im,5)==1
    D=4;%
end
if size(im,4)~=1 &&size(im,5)~=1
    D=[5,4];
end


V_mat=cell(1,length(D));
for k=1:length(D)
    m = D(k);
    Sm = tenmat(S,m);
    Sm = Sm.data;
    V_mat{k} = svd(Sm,'econ');	
end


