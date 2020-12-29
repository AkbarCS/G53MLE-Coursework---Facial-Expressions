function [I, sum_pn] = calculateI(targets)

num_pos = size(find(targets == 1),1);
num_neg = size(find(targets == 0),1);

sum_pn = num_pos + num_neg;

pos_prop = num_pos / sum_pn;
neg_prop = num_neg / sum_pn;

if (pos_prop == 0)
    pos = 0;
else
    pos = pos_prop * log2(pos_prop);
end

if (neg_prop == 0)
    neg = 0;
else
    neg = neg_prop * log2(neg_prop);
end

I = - (pos + neg);

end