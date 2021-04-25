function [p_best,q_best]=find_index_new(x,y)

if iscell(x) || iscell(y)
    [row,col]=size(x);
    x_cor=cell(row,col);
    y_cor=cell(row,col);
    for j=1:length(x)
        x_cor{j}=x{j}';
        y_cor{j}=y{j}';
    end
    x_cor_mat=cell2mat(x_cor);
    y_cor_mat=cell2mat(y_cor);
    xy=[x_cor_mat',y_cor_mat'];
[C,~,ic] = unique(xy,'rows');
tbl = tabulate(ic);
[~,ind] = max(tbl(:,2));
result = C(ind,:);
p_best=result(1);
q_best=result(2);
end

end

