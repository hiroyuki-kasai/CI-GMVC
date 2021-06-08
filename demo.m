%
% demo script of "Consistency-aware and Inconsistency-aware Graph-based Multi-view Clustering"
%

clc  
close all 
clear

%% settings
runtimes = 1;       % run-times on each dataset, default: 1
choice_graph = 2;   % 1: 'Complete', and 2: 'k-nearest'
choice_metric = 4;  % 1: 'Binary', 2: 'Cosine', 3: 'Gaussina-kernel', and 4: 'GBS-method'
lambda = 1;         %  initial parameter, which is tuned automatically
verbose = 3;

% For inconsistency part
beta = 1e-12;
gamma = 1e-5;
    

%% load dataset
load('datasets/100leaves.mat');
X = data;
num = size(X{1}, 2); % number of instances
view_num = length(X); % number of views

% normalization: Z-score
for i = 1:view_num
    X{i} = zscore(X{i});    
end        
do_normalize = 0; 

y0 = truelabel{1};
class_num = length(unique(truelabel{1}));


%% initialize
S0 = [];
max_iter = 100;

%% perform GBS
[F, y, U, S0, evs, infos_gbs] = ci_gmvc(X, class_num, choice_graph, choice_metric, lambda, do_normalize, 0, 0, max_iter, S0, verbose);


%% perform CI-GMVC
[F, y, U, S0, evs, infos_cigmvc] = ci_gmvc(X, class_num, choice_graph, choice_metric, lambda, do_normalize, beta, gamma, max_iter, S0, verbose);


%% plot
fs = 20;
% iter vs. cost without reg
figure;
semilogy([infos_gbs.iter], [infos_gbs.cost_noreg], '-O','Color','blue','linewidth', 2.0); hold on
semilogy([infos_cigmvc.iter], [infos_cigmvc.cost_noreg], '-O','Color','red','linewidth', 2.0); hold off
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'Iterations','FontSize',fs);
str = '$$ \sum^{V}_{v=1} \alpha_v || {\bf U} - {\bf S}_v || ^{2}_{F} $$';
ylabel(ax1,str,'Interpreter','latex'); 
legend({'GBS', 'CI-GMVC'},'Location','best');        
hold off
grid on


% time vs. cost without reg
figure;
semilogy([infos_gbs.time], [infos_gbs.cost_noreg], '-O','Color','blue','linewidth', 2.0); hold on
semilogy([infos_cigmvc.time], [infos_cigmvc.cost_noreg], '-O','Color','red','linewidth', 2.0); hold on
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'Times [sec]','FontSize',fs);
str = '$$ \sum^{V}_{v=1} \alpha_v || {\bf U} - {\bf S}_v || ^{2}_{F} $$';
ylabel(ax1,str,'Interpreter','latex');   
legend({'GBS', 'CI-GMVC'},'Location','best');  
hold off
grid on
    

