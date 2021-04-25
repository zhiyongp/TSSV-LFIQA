%%% txw_IQA������ �����͹����۽��������DMOSֵ�����ܼ�����������۵�4��ָ���Լ���ͼ��
%%% �ɿ͹�����ֵ������ֵ����������ֵvs�͹�����ֵ���Լ���DMOS VS DMOSp��ɢ��ͼ�������CC��SROCC��OR��RMSE 4������ָ��ֵ��
%%% ʹ��ʱ����ǵý�nihe,logistics������txw_IQA�ŵ�һ��Ŀ¼�£������ݾ�������޸�X0ֵ��

function [ pearson_cc, spearman_srocc, rmse ] = svr_IQA( obj_result, dmos)
% obj_result:�͹�ģ�ͼ����ԭʼ��û����Ϲ��Ŀ͹�ֵ��Ϊһ���������ݣ�
% dmos:��������ֵ��������DMOSֵ��
% dmos_std:��������ֵ�ı�׼��ڼ���ORֵ��ʱ����õ����������Ҫ����ORֵ�Ļ�����0�����Ȼ��orֵ��ȡ���ɡ�
% distortion_type: ָ��ʧ�����ͣ�������Ҫ�����ڻ�ͼʱ������ʾ�õġ�
% ����ɢ��ͼ���������κο�
%% ��ȡdmosp [ʵ�����ǶԿ͹�ֵ���� �Ĳ���\�����logistic������������ϵõ�dmospֵ]
xdata = obj_result;
ydata = dmos;

% X0 = [ max( dmos ), min( dmos ), mean( obj_result ), 1 ]; %��С�����������ʼֵX0���Ĳ���
X0 = [ max( dmos ), min(dmos ), median( obj_result ),mean(dmos),1 ]; %��С�����������ʼֵX0,�����

x = lsqcurvefit( @nihe, X0, xdata, ydata ); %��С��������

dmosp = nihe(x,xdata); %���ǿ͹�ֵ���Ĳ�����Ϻ��dmospֵ,�ο������ܿ����Ĳ�ʿ����72ҳ���������õ�����140������Ǵ���ģ�


%% ��ʼ����ָ��ļ���
%% CCϵ��
pearson_cc = corr( [ dmos, dmosp] );
pearson_cc = pearson_cc( 2 );

%% SROCCϵ��
spearman_srocc = corr( [ dmos, obj_result], 'type', 'Spearman' );
spearman_srocc = spearman_srocc( 2 );


%% KROCC(OR�Ѿ���ʱ�ˣ���������ʹ��)
% kendall_krocc = corr( [ dmos, obj_result], 'type', 'kendall' );
% kendall_krocc = kendall_krocc( 2 );
%% RMSEϵ��
rmse = sqrt( mean( ( dmosp - dmos ).^2 ) );

%% ��ͼ

% {
% DMOS vs DMOSp ���ͼ�����ɢ��Խ�ӽ��Խ��ߣ�˵������ģ��Խ�á�
% figure;
% plot( dmosp, dmos, 'b.' );
% title( 'distortion_type' ); xlabel( 'DMOSp' ); ylabel( 'DMOS' );
% hold on;
% plot( [ 0:1:max(dmos)+5 ], [ 0:1:max(dmos)+5 ], 'r-', 'LineWidth', 3); %����ֻ�Ƕ໭����һ���Խ��ߣ���Ϊ�ο�
%}
%

% DMOS vs �͹�ֵQ ���ͼ�����Խ�ӽ�������ߣ�˵��ģ��Խ�á�
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
%�ⲿ�֣���������ͬʧ�������ò�ͬ��ɫ��ע��ͬһ��ɢ��ͼ�еĴ��룬������ֵ���ݾ����仯����Ҫ���Ͻ�figureע�͵���
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



%% �Ĳ������������Ϻ���
function F = nihe(x,xdata)
% F = x(2) + (x(1)-x(2))./( 1 + exp( -( xdata - x(3))/abs(x(4)) ) );  %�Ĳ���logistic ���
F = x(1)*logistics( x(2),( xdata - x(3) ) ) + x(4)*xdata +x(5);  %��Ҫ�õ�logistics�����������logistic ��ϣ����� ��A statistical evaluation of recent������
% F = x(1)*xdata.^3 + x(2)*xdata.^2 + x(3)*xdata + x(4);  %�Ĳ�������ʽ ���
end

%% logistics function
function F = logistics( t,x )
F = 1/2 - 1./( 1 + exp(t*x) );
end