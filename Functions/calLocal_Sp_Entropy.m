function H_local_sp = calLocal_Sp_Entropy(img)
%Local spectral entropy of grayscale image
fun1=@(x)fecal(x);
H_local_sp=blkproc(img,[8 8] ,fun1);