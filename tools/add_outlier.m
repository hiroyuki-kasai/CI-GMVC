function [ V, Ro ] = add_outlier(rho, F, N, Vo, val_ratio)

    ori_max = max(max(Vo));
    ori_min = min(min(Vo));  
    
    max_val = floor(val_ratio * ori_max);
    if max_val < 1
        max_val = 1;
    end
    
    dense = rho * F;
    Ro = zeros(F,N);
    
    for i = 1 : N
        n_before = 0;

            for f = 1 : dense
                c = randi(F);
                 Ro(c,i) = randi([0, max_val]);
                 n = nnz(Ro(:,i));
                 if n_before == n
                     f = f - 1;
                 end
                 n_before = n;
            end
    end
    V = Vo + Ro;
    
    V = max(V, 0);
    %V = min(V,50);
end

