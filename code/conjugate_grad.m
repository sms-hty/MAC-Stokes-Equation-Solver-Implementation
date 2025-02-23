function [U_delta, V_delta, iter_num] = conjugate_grad(Residual_U, Residual_V, max_iter, U0, V0, tol, N, device)
    %默认U0 V0是0矩阵
    U_delta = U0;
    V_delta = V0;
%     [A_U, A_V] = apply_A(U_delta, V_delta, N, device);
    Res_U = Residual_U;% - A_U;
    Res_V = Residual_V;% - A_V;
    P_u = Res_U;
    P_v = Res_V;
    rou_u = norm(Res_U, 'fro')^2;
    rou_v = norm(Res_V, 'fro')^2;
%     rou = sum(Res_U .^2, "all") + sum(Res_V .^2, "all");
    norm_res = sqrt(rou_u + rou_v);
%     toc
    for iter = 1:max_iter
        iter_num = iter;
        [A_pu, A_pv] = apply_A(P_u, P_v, N, device);

        alpha_u = rou_u / sum(P_u .* A_pu, "all");
        alpha_v = rou_v / sum(P_v .* A_pv, "all");
        U_delta = U_delta + alpha_u * P_u; 
        V_delta = V_delta + alpha_v * P_v;
        Res_U = Res_U - alpha_u * A_pu;
        Res_V = Res_V - alpha_v * A_pv;
        rou_u_new = norm(Res_U, 'fro')^2;
        rou_v_new = norm(Res_V, 'fro')^2;
        
        if sqrt(rou_u_new + rou_v_new) / norm_res < tol
            break;
        end

        beta_u = rou_u_new / rou_u;
        beta_v = rou_v_new / rou_v;
        P_u = Res_U + beta_u * P_u;
        P_v = Res_V + beta_v * P_v;
        rou_u = rou_u_new;
        rou_v = rou_v_new;
        
    end
%     toc
end