function [U, V, P, ite] = Inexact_Uzawa_iteration(F_U, F_V, N, alpha, v1, v2, PCG_ite_max, Vcycle_ite_max, Vcycle_error, tau, bottom, epsilon, device)
    ite = 0;
    if device == 'gpu'
        U = gpuArray.zeros(N+1, N);
        V = gpuArray.zeros(N, N+1);
        P = gpuArray.zeros(N, N);
        F_U = gpuArray(F_U);
        F_V = gpuArray(F_V);      
    else
        U = zeros(N+1, N);
        V = zeros(N, N+1);
        P = zeros(N, N);
    end
    Res_U = F_U;
    Res_V = F_V;
    r0 = sqrt(sum(F_U.^2, 'all') + sum(F_V.^2, 'all'));
    while true
        ite = ite + 1;
        [U, V, pcg_ite, cycle_ite, cycle_time] = pre_conjugate_grad(Res_U, Res_V, N, PCG_ite_max, Vcycle_ite_max, Vcycle_error, tau, bottom, v1, v2, epsilon, device);
        fprintf("PCG ite=%d \n",pcg_ite);
        fprintf("Vcycle循环次数:%d\n", cycle_ite);
%         fprintf("每次Vcycle循环用时：")
%         disp(cycle_time);
        P = P + alpha * apply_Btrans(U, V, N);
        [A_U, A_V] = apply_A(U, V, N, device);
        [B_Pu, B_Pv] = apply_B(P, N, device);
        Res_U = F_U - B_Pu;
        Res_V = F_V - B_Pv;
        rh = sqrt(sum((Res_U - A_U) .^2, 'all') + sum((Res_V - A_V) .^2, 'all') + sum(apply_Btrans(U, V, N) .^2, 'all'));
        if rh < 1e-8 * r0
            break;
        end
    end
end