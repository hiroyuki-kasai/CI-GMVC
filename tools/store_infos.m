%function [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time)
function [infos] = store_infos(infos, iter, cost, cost_noreg, elapsed_time)
% Function to store statistic information
%
% Inputs:
%       problem         function (cost/grad/hess)
%       w               solution 
%       options         options
%       infos           struct to store statistic information
%       epoch           number of outer iteration
%       grad_calc_count number of calclations of gradients
%       elapsed_time    elapsed time from the begining
% Output:
%       infos           updated struct to store statistic information
%       f_val           cost function value
%       outgap          optimality gap
%       grad            gradient
%       gnorm           norm of gradient
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Sep. 25, 2017
% Modified by H.Kasai on Mar. 27, 2017


    if ~iter
        
        infos.iter = iter;
        infos.time = 0;    
        infos.cost = cost;
        infos.cost_noreg = cost_noreg;
        
    else
        
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];
        
        infos.cost = [infos.cost cost];
        infos.cost_noreg = [infos.cost_noreg cost_noreg];        
        
    end

end
