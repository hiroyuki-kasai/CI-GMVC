function [F, y, U, S0, evs, infos] = ci_gmvc(X, class_num, choice_graph, choice_metric, lambda, do_normalize, beta, gamma, max_iter, S0, verbose)
% Code for Consistency-aware and Inconsistency-aware Graph-based Multi-view Clustering
%
% Inputs:
%       X               multi-view dataset, each cell corresponds to a view, each column corresponds to a data point     
%       class_num       number of classes
%       choice_graph    1: 'Complete', and 2: 'k-nearest'
%       choice_metric   1: 'Binary', 2: 'Cosine', 3: 'Gaussina-kernel', and 4: 'GBS-method'
%       lambda          initial parameter, but it is automatically calculated
%       do_normalize    flag for normalizing data
%       beta            regularization parameter of inconsistent part
%                       to avoid large value within a view
%       gamma           regularization parameter of inconsistent part 
%                       to avoid large value between two difference views
%       S0              initial SIG matrix
%       verbose         flag for verbosity        

% Output:
%       F               embedding matrix
%       y               final clustering result, i.e., cluster indicator vector
%       U               learned unified matrix
%       S0              constructed SIG matrix, each row corresponds to a data point
%       evs             eigenvalues of learned graph Laplacian in the iterations
%       infos           information
%
% Reference:
%       Mitsuhiko Horie and Hiroyuki Kasai,
%       Consistency-aware and Inconsistency-aware Graph-based Multi-view Clustering
%       EUSIPCO, 2020.
%
%
% This file is originally generated from two works below:
%
%       Hao Wang, Yan Yang, Bing Liu, Hamido Fujita
%       A Study of Graph-based System for Multi-view Clustering
%       Knowledge-Based Systems, 2019
%       https://github.com/cswanghao/gbs
%
%       Youwei Liang, Dong Huang, and Chang-Dong Wang. Consistency Meets 
%       Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering
%       IEEE International Conference on Data Mining(ICDM), 2019
%       https://github.com/youweiliang/ConsistentGraphLearning
%
%
% Created by M. Horie and H.Kasai on Feb. 07, 2020
% Modified by H.Kasai on May 29, 2021

    sample_num = size(X{1},2); % number of samples
    view_num = length(X); % number of views
    
    zr = 1e-10;
    islocal = 1; % default: only update the similarities of neighbors if islocal=1
    
    if isempty(choice_graph)
        choice_graph = 2; % suggest using k-nearest graph
    end
    
    if isempty(choice_metric)
        choice_metric = 4; % suggest using our method
    end
    
    if isempty(lambda)
        lambda = 1;
    end
    
    if isempty(do_normalize)
        do_normalize = 1;
    end
    

    mode = method_info_mod(choice_graph, choice_metric, beta, gamma, verbose);    

    
    %% normalization: Z-score
    if do_normalize == 1
        for i = 1 : view_num
            X{i} = zscore(X{i});    
        end
    end 

    
    %% Constructing the SIG matrices
    pn = 15; % pn: number of adaptive neighbours
    options = [];
    options.k = 5;

    if isempty(S0)
        S0 = cell(1,view_num);
        for i = 1 : view_num
            if 1 == choice_graph % complete graph
                options.k = 0;
                if 1 == choice_metric
                    options.WeightMode = 'Binary';
                    S0{i} = constructS_KNG(X{i}', options);
                elseif 2 == choice_metric
                    options.WeightMode = 'Cosine';
                    S0{i} = constructS_KNG(X{i}', options);
                elseif 3 == choice_metric
                    options.WeightMode = 'HeatKernel';
                    S0{i} = constructS_KNG(X{i}', options);
                else
                    if verbose > 0
                        error('Invalid input: check choice_metric');
                    end
                end
            elseif 2 == choice_graph % k-nearest graph
                if 1 == choice_metric
                    options.WeightMode = 'Binary';
                    S0{i} = constructS_KNG(X{i}', options);
                elseif 2 == choice_metric
                    options.WeightMode = 'Cosine';
                    S0{i} = constructS_KNG(X{i}', options);
                elseif 3 == choice_metric
                    options.WeightMode = 'HeatKernel';
                    S0{i} = constructS_KNG(X{i}', options);
                elseif 4 == choice_metric
                    [S0{i}, distX_i] = constructS_PNG(X{i}, pn, 0);
                else
                    if verbose > 0
                        error('Invalid input: check choice_metric');
                    end
                end
            else
                if verbose > 0
                    error('Invalid input: check choice_graph');
                end
            end
        end
    end
    
    
    % initialize U, F and w
    U0 = zeros(sample_num);
    for i = 1 : view_num
        U0 = U0 + S0{i};
    end
    U0 = U0/view_num;
    for j = 1 : sample_num
        d_sum = sum(U0(j,:));
        if d_sum == 0
            d_sum = eps;
        end
        U0(j,:) = U0(j,:)/d_sum;
    end
    U = (U0+U0')/2;

    D = diag(sum(U));
    L = D - U;
    [F, ~, evs] = eig1(L, class_num, 0);

    w = ones(1,view_num)/view_num;
    
    % initialize A
    A = cell(1,view_num);
    
    if gamma ~= 0 && beta ~= 0   
        
        S_ave = zeros(size(S0{i}));
        for i = 1 : view_num
            S_ave = S_ave + S0{i};
        end  
        S_ave = S_ave / view_num;
        
        for i = 1 : view_num
            A{i} = S_ave;
            A{i} = S0{i};
        end  
    else
        for i = 1 : view_num
            A{i} = S0{i};
        end          
    end
    
    H = cell(view_num,1);    
    B = gamma*ones(view_num) - diag(gamma*ones(1,view_num)) + diag(beta*ones(1,view_num));    
    
    % initialize commom_baS
    commom_baS = zeros(sample_num,sample_num);
    gamma_S = cell(view_num,1);
    beta_S = cell(view_num,1);
    
    for i = 1 : view_num
        gamma_S{i} = gamma*S0{i};
        beta_S{i} = beta*S0{i};
        commom_baS = commom_baS + gamma_S{i}; % same values in every iteration
    end    
    
    % initialize true_baS   
    true_baS = cell(view_num,1);    
    for i = 1 : view_num
        true_baS{i} = commom_baS - gamma_S{i} + beta_S{i}; % same values in every iteration
    end    
    
    infos = [];  
    total_elapsed_time = 0;
    
    iter = 0;
    [cost_noreg, cost] = calculate_cost_function(view_num, U, S0, A, w, F, L, lambda, B); 
    if verbose > 1
        fprintf('# %s: %d: cost_ori=%.5f, cost=%.5f\n', mode, iter, cost_noreg, cost);
    end
    [infos] = store_infos(infos, iter, cost, cost_noreg, 0);    

    
    %%  main loop
    for iter = 1:max_iter
        
        % set start time
        start_time = tic();
      
        % update W
        for v = 1 : view_num
            US = U - A{v};
            distUS = norm(US, 'fro')^2;
            if distUS == 0
                distUS = eps;
            end
            w(v) = 0.5/sqrt(distUS);
        end
        
        % update U
        dist = L2_distance_1(F',F');
        U = zeros(sample_num);
        for i=1 : sample_num
            idx = zeros();
            for v = 1 : view_num
                a0 = A{v}(i,:);
                idx = [idx,find(a0>0)];
            end

            idxa = unique(idx(2:end));
            
            if islocal == 1
                idxa0 = idxa;
            else
                idxa0 = 1 : sample_num;
            end
            
            for v = 1 : view_num
                a1 = A{v}(i,:);
                ai = a1(idxa0);
                di = dist(i,idxa0);
                mw = view_num*w(v);
                lmw = lambda/mw;
                q(v,:) = ai-0.5*lmw*di;
            end
            U(i,idxa0) = SloutionForP20(q,view_num);
            clear q;
        end

        % update F
        sU = U;
        sU = (sU+sU')/2;
        D = diag(sum(sU));
        L = D-sU;
        F_old = F;
        [F, ~, ev] = eig1(L, class_num, 0);
        evs(:,iter+1) = ev;
        
        % update lambda and the stopping criterion
        fn1 = sum(ev(1:class_num));
        fn2 = sum(ev(1:class_num+1));
        if fn1 > zr
            lambda = 2*lambda;
        elseif fn2 < zr
            lambda = lambda/2;
            F = F_old;
        else
            if verbose > 1
                fprintf('# %s: breaked at %d (lambda=%.2f)\n', mode, iter, lambda);
            end
            break;
        end
        
        % update A
        if gamma ~= 0 && beta ~= 0

            for i = 1 : view_num
                tmp = 2*w(i)*U + true_baS{i};
                H{i} = tmp(:);
            end
            
            vec_H = cat(2, H{:})';
            
            C = 2*diag(w) + B;
            
            if det(C) == 0
                solution = (pinv(C) * vec_H)';
                fprintf('------------')
            else
                solution = (C \ vec_H)';
            end
            
            solution(solution<0) = 0;

            for i = 1 : view_num
                temp = solution(:,i);
                oldA = A{i};
                A{i} = zeros(sample_num, sample_num);
                %A{i}(up_knn_idx) = temp;
                A{i} = reshape(temp, [sample_num sample_num]);
                A{i} = max(A{i}, A{i}');
                A{i} = min(S0{i}, A{i});
            end
        end
        
        % measure elapsed time
        total_elapsed_time = total_elapsed_time + toc(start_time);                
        
        % store infos
        [cost_noreg, cost] = calculate_cost_function(view_num, U, S0, A, w, F, L, lambda, B);
        if verbose > 1
            fprintf('# %s: %d: cost(w/o reg)=%.5f, cost=%.5f', mode, iter, cost_noreg, cost);
            
            if verbose > 2
                if gamma ~= 0 && beta ~= 0   
                    fprintf(', Enorm:');
                    for i = 1 : view_num
                        fprintf('[%d] %.2f, ', i, norm(S0{i} - A{i}, 'fro'));
                    end 
                    fprintf('\n');
                else
                    fprintf('\n');
                end
            else
                fprintf('\n');
            end

        end
        [infos] = store_infos(infos, iter, cost, cost_noreg, total_elapsed_time);                
        
    end
    
    %% generating the clustering result
    [clusternum, y] = graphconncomp(sparse(sU)); y = y';
    if clusternum ~= class_num
        if verbose > 0
            fprintf('Can not find the correct cluster number: %d\n', class_num)
        end
    end
    
    
end

function [cost_noreg, cost] = calculate_cost_function(view_num, U, S0, A, w, F, L, lambda, B)

    % calculate the objective value
    for v = 1 : view_num
        tempF(v) = w(v)*norm(U - A{v}, 'fro')^2;
    end

    fLf = F'*L*F;
    cost_noreg = sum(tempF);
    
    cost_noreg2 = cost_noreg + lambda*trace(fLf);
    
    tmp = 0;
    for w = 1 : view_num
        for v = 1 : view_num 
            tmp = tmp + B(w,v) * trace((S0{v}-A{v}) * (S0{w}-A{w})');
        end
    end     
    
    cost = cost_noreg2 + tmp;
        
end
    
    


