function [U, V, pcg_ite, cycle_ite, cycle_time] = pre_conjugate_grad(F_U, F_V, N, ite_max, Vcycle_ite_max, Vcycle_error, tau, bottom, v1, v2, epsilon, device)
    if device == 'gpu'
        U = gpuArray.zeros(N+1, N);
        V = gpuArray.zeros(N, N+1);
    else
        U = zeros(N+1, N);
        V = zeros(N, N+1);
    end
    pcg_ite = 0;
    cycle_ite = zeros(1, 1, 'int8');
    cycle_time = zeros(1, 1);
    norm_b = epsilon * sqrt(norm(F_U, 'fro')^2 + norm(F_V, 'fro')^2);

    time_1 = toc;
    [z_u, z_v, cycle_ite(1, 1)] = pre_V_cycle(F_U, F_V, v1, v2, Vcycle_ite_max, Vcycle_error, N, bottom, device);
    time_2 = toc;
    cycle_time(1, 1) = time_2 - time_1;

    Res_U = F_U;
    Res_V = F_V;
    P_u = z_u;
    P_v = z_v;
    rou_u = sum(Res_U .* P_u, 'all');
    rou_v = sum(Res_V .* P_v, 'all');

    for iter = 1:ite_max
        pcg_ite = pcg_ite + 1;
        [A_pu, A_pv] = apply_A(P_u, P_v, N, device);
        alpha_u = rou_u / sum(P_u .* A_pu, 'all');
        alpha_v = rou_v / sum(P_v .* A_pv, 'all');
        U = U + alpha_u * P_u;
        V = V + alpha_v * P_v;
        Res_U = Res_U - alpha_u * A_pu;
        Res_V = Res_V - alpha_v * A_pv;

        P = apply_Btrans(U, V, N);
        [A_U, A_V] = apply_A(U, V, N, device);
        
        r1 = sqrt(norm((F_U - A_U), 'fro')^2 + norm((F_V - A_V), 'fro'));
        r2 = tau *norm(P, 'fro');
        if r1 < r2 || iter == ite_max || r1 < norm_b
            break;
        end

        time_1 = toc;
        [z_u, z_v, cycle_ite(1, iter+1)] = pre_V_cycle(Res_U, Res_V, v1, v2, Vcycle_ite_max, Vcycle_error, N, bottom, device);
        time_2 = toc;
        cycle_time(1, iter+1) = time_2 - time_1;

        rou_u_new = sum(Res_U .* z_u, 'all');
        rou_v_new = sum(Res_V .* z_v, 'all');
        beta_u = rou_u_new / rou_u;
        beta_v = rou_v_new / rou_v;
        P_u = z_u + beta_u * P_u;
        P_v = z_v + beta_v * P_v;
        rou_u = rou_u_new;
        rou_v = rou_v_new;
    end
end