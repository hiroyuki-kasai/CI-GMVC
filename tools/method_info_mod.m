function mode = method_info_mod(choice_graph, choice_metric, beta, gamma, verbose)

    
    if gamma ~= 0 && beta ~= 0
        base_mode = 'CI-GMVC';
    else
        base_mode = 'GBS';        
    end

    if 1 == choice_graph % complete graph
        if 2 == choice_metric
            mode = sprintf('%s-CC', base_mode);
        elseif 3 == choice_metric
            mode = sprintf('%s-CG', base_mode);       
        end
    elseif 2 == choice_graph % k-nearest graph
        if 1 == choice_metric
            mode = sprintf('%s-KB', base_mode);               
        elseif 2 == choice_metric
            mode = sprintf('%s-KC', base_mode);                
        elseif 3 == choice_metric
            mode = sprintf('%s-KF', base_mode);                
        elseif 4 == choice_metric
            mode = 'GBS-KO';  
            mode = sprintf('%s-K0', base_mode);                
        end
    end
    
    if verbose > 2
        fprintf('Algorithm: %s\n', mode);
    end
        
end