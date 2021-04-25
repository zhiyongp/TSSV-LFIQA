function [y,max_x,min_x]=normalization_ML(x,low,high,MAX,MIN)
% min,max 0,1
% min,max -1,1
if (nargin==3)
    max_x=max(x(:));
    min_x=min(x(:));
    if low==0 &&high==1
        y=(x(:)-min_x)/(max_x-min_x+0.001);% +0.001避免出现NaN
    else
        y=(2*x(:)-min_x-max_x)/(max_x-min_x+0.001);% +0.001避免出现NaN
    end
else
    if low==0 &&high==1
        y=(x(:)-MIN)/(MAX-MIN+0.001);% +0.001避免出现NaN
    else
        y=(2*x(:)-MIN-MAX)/(MAX-MIN+0.001);% +0.001避免出现NaN
    end
    
end
end