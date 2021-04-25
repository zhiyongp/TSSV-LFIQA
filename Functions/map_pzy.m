function I = map_pzy(A)

Min = min(min(A));
Max = max(max(A));
if Max == Min
    I = ones(size(A));
else
    I = (A - Min) / (Max - Min);
end
end