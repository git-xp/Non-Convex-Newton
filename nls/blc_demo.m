function blc_demo(init, max_props)
%
% a demo of comparison among different methods for solving nonlinear least
% square problems: binary linear classification on a libsvm dataset: ijcnn1
%
% init:
%       0 -- zero initialization
%       1 -- all 1's initialization
%       2 -- random initialization
% max_props: maximum number of propagations
%
% written by Peng Xu, Fred Roosta, 6/8/2017, updated(2/8/2018)

if nargin < 2
    max_props = 1e8;
end
filename='ijcnn1'; % download data from here: 
addpath(genpath(pwd));
seed = 1234;
rng(seed);
[Y,X] = libsvmread('ijcnn1');
[Yt,Xt] = libsvmread('ijcnn1.t');
if isempty(X) || isempty(Y) || isempty(Xt) || isempty(Yt)
    fprintf('No data loaded!\n');
    return
end

[n,d] = size(X);
Y(Y~=1) = 0;
Yt(Yt~=1) = 0;
%%
problem.loss = @compute_blc_loss;
problem.grad = @compute_blc_gradient;
problem.hessian = @compute_blc_hessian_diag;
problem.get_gn = @compute_blc_gn_diag;
lambda = 0;
problem.lambda = lambda;

%%%%%%%%%%%%% Parameter setting %%%%%%%%%%%%%%%%%%
if init == 1
    w0  = ones(d,1);
    fprintf('all 1\''s initialization\n\n')
elseif init == 0
    w0 = zeros(d,1);
    fprintf('zero initialization\n\n');
else
    w0 = randn(d,1);
    fprintf('random initialization\n\n');
end
problem.w0 = w0;
uniform_factor = 5;
nonuniform_factor = 5;
max_delta = Inf;
delta = 10; % initial trust-region radius
min_sigma = 0;
sigma = 1e-4; % initial cubic regularization parameter

markers = {'k-','b-','r-','k-.','b-.','r-.','k--','b--','r--', 'g:'};
BFGS_History = 100;
eta = 1; %stepsize for GN and L-BFGS
%eta_gd = 1E-4; %stepsize for GD


lineWidth = 2;
legendLocation = 'bestoutside';
if init == 1
    init_type = 'all 1''s initialization';
    dir_name = ['./figs/blc_ones/',filename,'/'];
    dir_out = ['./figs/blc_ones/',filename,'/'];
elseif init == 0
    init_type = 'zero initialization';
    dir_name = ['./figs/blc_zeros/',filename,'/'];
    dir_out = ['./figs/blc_zeros/',filename,'/'];
else
    init_type = 'random initialization';
    dir_name = ['./figs/blc_randn/',filename,'/'];
    dir_out = ['./figs/blc_randn/',filename,'/'];
end
if ~exist(dir_name, 'dir')
    status = mkdir(dir_name);
end

uniform_sample_size = ceil(uniform_factor*n/100);%10*d;
rns_sample_size = ceil(nonuniform_factor*n/100); %2*d;


methods{1} = struct('name','TR Full','method','Newton-TR', 'hessian_size',n, 'step_size',1,...
    'max_props', max_props,'delta', delta, 'max_delta', max_delta,'solver','Steihaug');
methods{2} = struct('name',sprintf('TR Uniform (%g%%)', uniform_factor), ...
    'method','Uniform-TR', 'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'delta', delta,'max_delta', max_delta,'solver','Steihaug');
methods{3} = struct('name',sprintf('TR Non-Uniform (%g%%)', nonuniform_factor), ...
    'method','RNS-TR', 'hessian_size', rns_sample_size, 'step_size', 1, ...
    'max_props', max_props,'delta', delta,'max_delta', max_delta,'solver','Steihaug');
methods{4} = struct('name','ARC FULL','method','Newton-ARC', 'hessian_size',n, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma);
methods{5} = struct('name',sprintf('ARC Uniform (%g%%)', uniform_factor), ...
    'method','Uniform-ARC', 'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma);
methods{6} = struct('name',sprintf('ARC Non-Uniform (%g%%)', nonuniform_factor), ...
    'method','RNS-ARC', 'hessian_size', rns_sample_size, 'step_size', 1, ...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma);
methods{7} = struct('name', 'GN Full', 'method','GN','step_size', eta, ...
    'max_props', max_props, 'linesearch',true, 'alpha', 1E-4, 'beta', 0.5);
methods{8} = struct('name',sprintf('GN Uniform (%g%%)', uniform_factor), ...
    'method','Uniform-GN', 'hessian_size', uniform_sample_size,'step_size', eta, ...
    'max_props', max_props, 'linesearch',true, 'alpha', 1E-4, 'beta', 0.5);
methods{9} = struct('name',sprintf('GN Non-Uniform (%g%%)', nonuniform_factor), ...
    'method','RNS-GN', 'hessian_size', rns_sample_size,'step_size', eta, ...
    'max_props', max_props, 'linesearch',true, 'alpha', 1E-4, 'beta', 0.5);
methods{10} = struct('name', sprintf('LBFGS-%g', BFGS_History), 'method','LBFGS',...
    'L',BFGS_History, 'max_props', max_props, 'step_size', eta,'linesearch',true, 'alpha', 1E-4, 'beta', 0.5);



for i = 1:10
    rng(seed);
    disp([methods{i}.name, ':']);
    savename = sprintf([dir_name,'all_%s_%s.mat'],filename,methods{i}.name);
    if exist(savename,'file')
        load(savename,'result');
    else
        [w, result] = subsampled_blc(X,Y,problem, methods{i});
        tr_acc = blc_eval(X,Y, result.sol);
        result.tr_acc = tr_acc;
        te_acc = blc_eval(Xt,Yt, result.sol);
        result.te_acc = te_acc;
        save(savename,'result');
    end
    figure(1)
    semilogx(result.noProps, result.l, markers{i},'DisplayName', methods{i}.name, 'LineWidth', lineWidth ); hold on
    figure(2)
    semilogx(result.noProps,1-result.tr_acc, markers{i}, 'DisplayName', methods{i}.name, 'LineWidth', lineWidth ); hold on
    figure(3)
    semilogx(result.noProps,1-result.te_acc, markers{i}, 'DisplayName', methods{i}.name, 'LineWidth', lineWidth ); hold on
    
end


%%
figure(1)
grid('on')
set(gca, 'fontsize', 30)
hold off;
xlabel('# of Props')
ylabel('training loss')
xlim([0,2*max_props]);
title(sprintf('%s, %s', filename, init_type));
figure(2)
legend('location',legendLocation);
grid('on')
set(gca, 'fontsize', 30)
hold off;
xlabel('# of Props')
ylabel('train error')
xlim([0,2*max_props]);
title(sprintf('%s, %s', filename, init_type));
figure(3)
grid('on')
set(gca, 'fontsize', 30)
hold off;
xlabel('# of Props')
ylabel('test error')
title(sprintf('%s, %s', filename, init_type));
xlim([0,2*max_props]);

if ~exist(dir_out, 'dir')
    status = mkdir(dir_out);
end
saveas(figure(1),sprintf([dir_out,'all_%s_noProps_tr_loss'],filename),'fig');
saveas(figure(1),sprintf([dir_out,'all_%s_noProps_tr_loss'],filename),'epsc');
saveas(figure(2),sprintf([dir_out,'all_%s_legend'],filename),'fig');
saveas(figure(2),sprintf([dir_out,'all_%s_legend'],filename),'epsc');
saveas(figure(3),sprintf([dir_out,'all_%s_noProps_te_acc'],filename),'fig');
saveas(figure(3),sprintf([dir_out,'all_%s_noProps_te_acc'],filename),'epsc');



