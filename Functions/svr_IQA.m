%%% txw_IQA函数： 给出客观评价结果和主观DMOS值，就能计算出质量评价的4个指标以及画图。
%%% 由客观评价值与主观值画出（主观值vs客观评价值）以及（DMOS VS DMOSp）散点图，且求出CC、SROCC、OR、RMSE 4个评价指标值；
%%% 使用时，请记得将nihe,logistics函数跟txw_IQA放到一个目录下，并根据具体情况修改X0值。

function [ pearson_cc, spearman_srocc, rmse ] = svr_IQA( obj_result, dmos)
% obj_result:客观模型计算的原始的没有拟合过的客观值，为一个矩阵数据；
% dmos:主观评价值，即各类DMOS值；
% dmos_std:主观评价值的标准差，在计算OR值的时候会用到。如果不需要计算OR值的话，以0替代，然后or值不取即可。
% distortion_type: 指明失真类型，这里主要是用于画图时标题显示用的。
% 画黑散点图，适用于任何库
%% 求取dmosp [实际上是对客观值进行 四参数\五参数logistic函数非线性拟合得到dmosp值]
xdata = obj_result;
ydata = dmos;

% X0 = [ max( dmos ), min( dmos ), mean( obj_result ), 1 ]; %最小二乘问题的起始值X0，四参数
X0 = [ max( dmos ), min(dmos ), median( obj_result ),mean(dmos),1 ]; %最小二乘问题的起始值X0,五参数

x = lsqcurvefit( @nihe, X0, xdata, ydata ); %最小二乘问题

dmosp = nihe(x,xdata); %这是客观值经四参数拟合后的dmosp值,参考的是周俊明的博士论文72页（这里引用的文献140里可能是错误的）


%% 开始各类指标的计算
%% CC系数
pearson_cc = corr( [ dmos, dmosp] );
pearson_cc = pearson_cc( 2 );

%% SROCC系数
spearman_srocc = corr( [ dmos, obj_result], 'type', 'Spearman' );
spearman_srocc = spearman_srocc( 2 );


%% KROCC(OR已经过时了，现在无人使用)
% kendall_krocc = corr( [ dmos, obj_result], 'type', 'kendall' );
% kendall_krocc = kendall_krocc( 2 );
%% RMSE系数
rmse = sqrt( mean( ( dmosp - dmos ).^2 ) );

%% 画图

% {
% DMOS vs DMOSp 这个图像如果散点越接近对角线，说明评价模型越好。
% figure;
% plot( dmosp, dmos, 'b.' );
% title( 'distortion_type' ); xlabel( 'DMOSp' ); ylabel( 'DMOS' );
% hold on;
% plot( [ 0:1:max(dmos)+5 ], [ 0:1:max(dmos)+5 ], 'r-', 'LineWidth', 3); %这里只是多画出了一条对角线，作为参考
%}
%

% DMOS vs 客观值Q 这个图像如果越接近拟合曲线，说明模型越好。
% figure;
% plot( obj_result, dmos, 'b.','MarkerSize',13 );
% title( 'distortion_type' ); xlabel( 'Q' ); ylabel( 'DMOS' );
% hold on;
% [ new_obj, index ] = sort( obj_result, 1, 'ascend' );
% new_dmosp = zeros( size( dmosp ) );
% new_dmosp(:) = dmosp( index(:) );
% plot( new_obj, new_dmosp, 'r-', 'LineWidth', 2.5 );

%
%-----
%这部分，用来将不同失真类型用不同颜色标注在同一幅散点图中的代码，具体数值依据具体库变化。需要将上节figure注释掉。
%-----
% figure;
% plot( obj_result(1:80,1), dmos(1:80,1), 'co' );                  % JP2K
% hold on; plot( obj_result(81:160,1),  dmos(81:160,1), 'gx' );         % JPEG
% hold on; plot( obj_result(161:240,1), dmos(161:240,1), 'b+' );         % WN
% hold on; plot( obj_result(241:320,1), dmos(241:320,1), 'm*' );         % FF
% hold on; plot( obj_result(321:365,1), dmos(321:365,1), 'kd' );         % Blur
% 
% title( distortion_type ); xlabel( 'Q' ); ylabel( 'DMOS' );
% hold on;
% [ new_obj, index ] = sort( obj_result, 1, 'ascend' );
% new_dmosp = zeros( size( dmosp ) );
% new_dmosp(:) = dmosp( index(:) );
% plot( new_obj, new_dmosp, 'r-', 'LineWidth', 2.5 );
end



%% 四参数或五参数拟合函数
function F = nihe(x,xdata)
% F = x(2) + (x(1)-x(2))./( 1 + exp( -( xdata - x(3))/abs(x(4)) ) );  %四参数logistic 拟合
F = x(1)*logistics( x(2),( xdata - x(3) ) ) + x(4)*xdata +x(5);  %需要用到logistics函数，五参数logistic 拟合，文献 ：A statistical evaluation of recent。。。
% F = x(1)*xdata.^3 + x(2)*xdata.^2 + x(3)*xdata + x(4);  %四参数多项式 拟合
end

%% logistics function
function F = logistics( t,x )
F = 1/2 - 1./( 1 + exp(t*x) );
end
