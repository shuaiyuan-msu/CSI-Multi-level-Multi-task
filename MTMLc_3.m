function [W, funcVal] = MTMLc_3(XR, V, rho1,rho4, opts,initW)
% output W is R

if nargin <5
    opts = [];
end
% initialize options.
opts=init_opts(opts);
if ~isfield(opts, 'verbose') % print every iteration
    opts.verbose = 0;
end

[m,~] = size(V);
k = size(XR,2);
funcVal = [];
if nargin == 6
    W0 = initW;
else
    W0 = zeros(k,m);
end

bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W0;
Wz_old = W0;
t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval  (Ws);
    
    % line search
    while true
        [Wzp,l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho4 / gamma);
        Fzp = funVal_eval (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) + gamma/2 * norm(delta_Wzp, 'fro')^2;% eq(7)
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + rho4 * l1c_wzp);
%     funcVal = cat(1, funcVal, Fzp + rho4 * norm(Wzp,1));
    if (bFlag)
        if(opts.verbose)
            fprintf('\n The program terminates as the gradient step changes the solution(W) very small. \n');
        end
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    if(opts.verbose)
                        fprintf('\n The program terminates as the absolute change of funcVal is small. \n');
                    end
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    if(opts.verbose)
                        fprintf('\n The program terminates as the relative change of funcVal is small. \n');
                    end
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                if(opts.verbose)
                    fprintf('\n The program terminates as the funcVal is lower than threshold. \n');
                end
                break;
            end
        case 3
            if iter>=opts.maxIter
                if(opts.verbose)
                    fprintf('\n The program terminates as it reaches the maximum iteration. \n');
                end
                break;
            end
    end
    
    if(opts.verbose)
        fprintf('Iteration %8i| function value %12.4f \n',iter,funcVal(end));
    end
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end
W = Wzp;


% private functions
    function [z,l1_comp_val] = l1_projection (v, beta)
        % this projection calculates
        % argmin_z = \|z-v\|_2^2 + beta \|z\|_1
        % z: solution
        % l1_comp_val: value of l1 component (\|z\|_1)
        z = zeros(size(v));
        vp = v - beta/2;
        z (v> beta/2)  = vp(v> beta/2);
        vn = v + beta/2;
        z (v< -beta/2) = vn(v< -beta/2);
        
        l1_comp_val = sum(sum(abs(z)));
    end

    function [grad_W] = gradVal_eval(W)
        grad_W = - rho1*(XR'-W*V)*V';
    end


    function [funcVal] = funVal_eval (W)
        funcVal = 0.5 * rho1*norm (XR'-W*V,'fro')^2;
        
    end
end