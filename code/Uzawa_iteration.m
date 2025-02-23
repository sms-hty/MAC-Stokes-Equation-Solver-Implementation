function [U, V, P, t, CG_iter_num] = Uzawa_iteration(F_U, F_V, N, alpha, device)
    t = 0;
    if device == 'gpu'
        U = gpuArray.zeros(N+1, N);
        V = gpuArray.zeros(N, N+1);
        P = gpuArray.zeros(N, N);
        zeros_U = gpuArray.zeros(N+1, N);
        zeros_V = gpuArray.zeros(N, N+1);
        F_U = gpuArray(F_U);
        F_V = gpuArray(F_V);
    else
        U = zeros(N+1, N);
        V = zeros(N, N+1);
        P = zeros(N, N);
        zeros_U = zeros(N+1, N);
        zeros_V = zeros(N, N+1);
    end
    CG_iter_num = zeros(1, 1);

    r0 = sqrt(norm(F_U, 'fro')^2 + norm(F_V, 'fro')^2);
    while true
        t = t + 1;
        
        [B_Pu, B_Pv] = apply_B(P, N, device);
        Res_U = F_U - B_Pu;
        Res_V = F_V - B_Pv;
        
        [U, V, CG_iter_num(1, t)] = conjugate_grad(Res_U, Res_V, 3*N, zeros_U, zeros_V, 1e-10, N, device);
        P = P + alpha * apply_Btrans(U, V, N); 

        [A_U, A_V] = apply_A(U, V, N, device);
        [B_Pu, B_Pv] = apply_B(P, N, device);
        U1 = A_U + B_Pu;
        V1 = A_V + B_Pv;
        P1 = apply_Btrans(U, V, N);
        rh=sqrt(norm(U1-F_U, 'fro')^2+norm(V1-F_V, 'fro')^2+norm(P1, 'fro')^2);
        if rh/r0<1e-8
            break
        end
    end

end