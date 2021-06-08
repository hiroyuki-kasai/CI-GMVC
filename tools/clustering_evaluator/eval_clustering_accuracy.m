function [accuracy] = eval_clustering_accuracy(gnd, labels)

%   gnd: ground truth label (Nx1)


    if size(gnd,2) ~= 1
        gnd = gnd';
    end
    if size(labels, 2) ~= 1
        labels = labels';
    end

    accuracy = [];
    
    best_nmi = 0;
    best_idx = 1;
    
    eval_num = size(labels, 2);

    for i = 1 : eval_num
        label = labels(:, i);
        %mi_array(i) = MutualInfo(gnd, label);
        purity_array(i) = calc_purity(gnd, label);
        nmi_array(i) = calc_nmi(gnd, label);
        %nmi_array(i) = compute_nmi(gnd, label);
        [f_va_arrayl(i), accuracy.precision, accuracy.recall] = compute_f(gnd,label);
        C = bestMap(gnd,label);
        acc_array(i) = length(find(gnd == C))/length(gnd);
        
        if nmi_array(i) >= best_nmi
            best_nmi = nmi_array(i);
            best_idx = i;
        end
    end
    
    % store
    %accuracy.mi     = mi_array(best_idx);
    accuracy.purity = purity_array(best_idx);
    accuracy.nmi    = nmi_array(best_idx);
    accuracy.f_val  = f_va_arrayl(best_idx);
    accuracy.acc    = acc_array(best_idx);
end

