function H_local = calLocalEntropy(img)
%%Local entropy of grayscale image
fun0=@(x)secal(x);
H_local=blkproc(img,[8 8] ,fun0);