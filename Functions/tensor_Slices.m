function [feature_map]=tensor_Slices(im)
% Function of Tensor decomposition
% S = HxWxCxVxU
S=tensor(im);
Hei =size(S.data,1);
Wid =size(S.data,2);

if ndims(im)==3
    D=[3];%Channels that need dimensionality reduction
    mlr=[Hei,Wid,9];
else
    if size(im,4)==1
        D=[5,3];
        mlr=[Hei,Wid,3,1,9];
        
    end
    if size(im,5)==1
        D=[4,3];
        mlr=[Hei,Wid,3,9,1];
    end
end



U=cell(1,ndims(S.data));
for k=1:length(D)
    m = D(k);
    Sm = tenmat(S,m);
    Sm = Sm.data;
    [Q,~,~] = svd(Sm,'econ');
    U{m} = Q(:,1:min(size(Q,2), mlr(m)));
    % Project onto dominant subspace.
    C = ttm(S, U, m, 't');%n-mode product
    S=C;
end
feature_map=squeeze(C.data);




