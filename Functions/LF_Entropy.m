function Feature = LF_Entropy( img_file,options )
%%% Local entropy includes spatial entropy and spectral entropy

%% Spatial Entropy
if options==1
    emat=calLocalEntropy(img_file);
    sort_t = sort(emat(:),'ascend');
    len = length(sort_t);
    weight=[0.2 0.8];
    t=sort_t(ceil(len*weight(1)):ceil(len*weight(2)));
    mu= mean(t);%Mean 
    ske=skewness(sort_t);%Skewness
%     kur =kurtosis(sort_t); %Kurtosis
    Feature=[mu,ske];
    
end

%% Spectral entropy
if options==2
    femat=calLocal_Sp_Entropy(img_file);
    sort_t_f = sort(femat(:),'ascend');
    len_f = length(sort_t_f);
    weight=[0.2 0.8];
    t_f=sort_t_f(ceil(len_f*weight(1)):ceil(len_f*weight(2)));
    mu_f= mean(t_f);%Mean 
    ske_f=skewness(sort_t_f);%Skewness
%     kur_f =kurtosis(sort_t_f); %Kurtosis
    Feature=[mu_f,ske_f];
end

end

