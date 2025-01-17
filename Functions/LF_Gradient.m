function [GM, RO, RM] = LF_Gradient( im )
%%% Calculate local gradient
% compute the RO RM and GM map
% the gradient magnitude��GM����the relative gradient orientation��RO��
%the relative gradient magnitude��RM����



sigma = 0.5;
[im_Dx,im_Dy] = gaussian_derivative(im,sigma);% compute the basic gradient maps in the horizontal x and vertical y directions

aveKernel = fspecial('average', 3);
eim_Dx = conv2(im_Dx, aveKernel,'same');
eim_Dy = conv2(im_Dy, aveKernel,'same');% compute the average directional derivative

im_D=atan(eim_Dx./(eim_Dy));
im_D(eim_Dy==0)=pi/2;

RO=atan(im_Dx./(im_Dy));
RO(im_Dy==0)=pi/2;
RO=RO-im_D; % compute RO

GM=sqrt(im_Dx.^2+im_Dy.^2); % compute GM

RM=sqrt((im_Dx-eim_Dx).^2+(im_Dy-eim_Dy).^2); %compute RM




    function [gx,gy] = gaussian_derivative(imd,sigma) % comput the gaussian derivative
        window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
        winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
        % winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
        winy=winx';
        gx = filter2(winx,imd,'valid');
        gy = filter2(winy,imd,'valid');
    end

end

