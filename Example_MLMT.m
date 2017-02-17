clear;clc;close;
addpath('../MLMT/functions/'); % load function
addpath('../MLMT/functions/others');
%% Create dataset 
[dataL,dataR] = CreateSyntheticdataMTML();
% Keep related variables
dataL = [dataL(:,1:2),dataL(:,end),dataL(:,3:end-1)]; % 1-lakeid, 2-regionid, 3-response, 4-end-predictor

% normalize the data 
tmp = dataL(:,3:end);
m = mean(tmp);s = std(tmp);
tmp = (tmp - repmat(m,size(tmp,1),1))./repmat(s,size(tmp,1),1);
dataL = [dataL(:,1:2),tmp];

% Combine region data 
Eduid = unique(dataL(:,2));
tmp = dataR;
m = mean(tmp);s = std(tmp);
tmp = (tmp - repmat(m,size(tmp,1),1))./repmat(s,size(tmp,1),1);
dataR = [Eduid,tmp]; % regionid,2-end predictor;

% generate trn tst index
trnrate = 2/3;vadrate = 1/2;ROUND = 1;method = 1;
[Trnidx,Tstidx,TrnNum,TstNum] = GenerateTrnTstIdx(dataL(:,[1,2,3]),trnrate,ROUND);%1-lakeid,2-eduid,3-reponse
LatLonL = [dataL(:,1:2),dataL(:,2),dataL(:,2)]; % 1-lakeid,2-eduid,3-lat,4-lon;

% add column of ones 
dataL = [dataL(:,[1,2]),dataL(:,3),ones(size(dataL,1),1),dataL(:,4:end)]; 

[Xtrn, Ytrn, Xtst, Ytst] = SplitTrnTst(dataL,Trnidx ,Tstidx);% data:1-lakeid, 2-eduid, 3-reponse, 4-end predictor
LatLonR = repmat(dataR(:,1),1,3);% 1-eduid,2-lat,3-lon;
% add dummy variable 
dataR = [ones(size(dataR,1),1),dataR(:,2:end)];
clear lakedata regiondata m s tmp;
%%  Initialization                                                          
result = cell(0);result{1,1} = 'obj func'; result{1,2} = 'runtime'; result{1,3} = 'rmse';result{1,4} = 'r2';
result{1,5} = 'rmse per region';result{1,6} = 'r2 per region';result{1,7} = 'ypred';result{1,8} = 'yreal';
result{1,9} = 'model w';result{1,10} = 'best param';result{1,11} = 'perform_mat';result{1,12} = 'funcFval';result{1,13} = 'model G';
higher_better = false;  % rmse is lower the better.
param_range = [0.1,1];

% optimization options
opts.init = 2;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.
opts.maxIter = 500; % maximum iteration number of optimization.
opts.verbose = 0;
opts.OutermaxIter = 100;

data_stl = zeros(size(dataL,1),size(dataL,2)+size(dataR(:,2:end),2));
for i = 1: size(Eduid,1)
    data_stl(dataL(:,2) == Eduid(i),:) = [dataL(dataL(:,2)==Eduid(i),:),repmat(dataR(i,2:end),sum(dataL(:,2)==Eduid(i)),1)];
end

%% Run Global-L
tic;
method = method+1;
obj_func_str = 'STL-global noregion';
[best_param, perform_mat] = TuneParam_lasso2(param_range,dataL,Trnidx,vadrate);
w = lasso(dataL(Trnidx,4:end),dataL(Trnidx,3),'lambda',best_param);
ypred = dataL(Tstidx,4:end)*w;
y = dataL(Tstidx,3);
[r2,rmse] = rsquare(y,ypred);

rmseall= zeros(size(Eduid)); r2all = rmseall;
id_tmp = dataL(Tstidx,2);
for t = 1: size(Eduid,1);
    y_pred_t = ypred(id_tmp == Eduid(t));
    y_t = y(id_tmp ==Eduid(t));
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = y;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear y ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w id_tmp t;
%% Run Global-LR
tic;
method = 1;
obj_func_str = 'STL-global model';
[best_param, perform_mat] = TuneParam_lasso2(param_range,data_stl,Trnidx,vadrate);
w = lasso(data_stl(Trnidx,4:end),data_stl(Trnidx,3),'lambda',best_param);
ypred = data_stl(Tstidx,4:end)*w;
y = data_stl(Tstidx,3);
[r2,rmse] = rsquare(y,ypred);

rmseall= zeros(size(Eduid)); r2all = rmseall;
id_tmp = dataL(Tstidx,2);
for t = 1: size(Eduid,1);
    y_pred_t = ypred(id_tmp == Eduid(t));
    y_t = y(id_tmp ==Eduid(t));
    [r2all(t),rmseall(t)] = rsquare(y_t,y_pred_t);
end

result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = y;result{method+1,9} = w;result{method+1,10} = best_param;result{method+1,11} = perform_mat;
clear y ypred y_pred_t y_t r2 rmse r2all rmseall best_param perform_mat w id_tmp t;
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval;
%% Two level methods
%% Run MLM
tic;
method = method+1;
obj_func_str = 'MTMLa';
eval_func_str = 'eval_rmse2_MTMLa';
[best_param, perform_mat] = TuneParam_MTMLa...
    (obj_func_str, opts, param_range, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
% build model using the optimal parameter
[G,Fval] = MTMLa(Xtrn, Ytrn,dataR, best_param, opts);
[rmse,rmseall,r2,r2all,ypred,yreal]= eval_rmse2_MTMLa(Xtst,Ytst,G,dataR);
for i = 1: length(Xtrn)
    W(:,i) = G'*dataR(i,:)';
end
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;result{method+1,13} = G;

%% MTMLc
tic;
method = method+1;
obj_func_str = 'MTMLc';
eval_func_str = 'eval_rmse';
rho1 = 5;
rho2 = param_range;
rho3 = param_range;
rho4 = param_range;
rho5 = param_range;
param_set = combvec(rho1,rho2,rho3,rho4,rho5)';
[best_param, perform_mat] = TuneParam_MTMLc...
    (obj_func_str, opts, param_set, eval_func_str, higher_better,dataL,Trnidx,vadrate,dataR); % dataL: 1-lakeid, 2-eduid, 3-response, 4-end
[U,V,R,Fval,W] = MTMLc2(Xtrn, Ytrn, dataR,best_param(1),best_param(2),best_param(3),...
    best_param(4),best_param(5),opts);
[rmse1,rmseall,r2,r2all,ypred,yreal]= eval_rmse2(Xtst,Ytst,W);
toc
result{method+1,1} = obj_func_str; result{method+1,2} = toc; result{method+1,3} = rmse1;result{method+1,4} = r2;result{method+1,5} = rmseall;result{method+1,6} = r2all;
result{method+1,7} = ypred;result{method+1,8} = yreal;result{method+1,9} = W;result{method+1,10} = best_param;result{method+1,11} = perform_mat;result{method+1,12} = Fval;
result{method+1,14} = U;result{method+1,15} = V;result{method+1,16} = R;result{method+1,13} = (U*R'*inv(R*R'))';
clear yreal ypred r2 rmse r2all rmseall best_param perform_mat W Fval G U V R i rho1 rho2 rho3 rho4 rho5 myparam;

