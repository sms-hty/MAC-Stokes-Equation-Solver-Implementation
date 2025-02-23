function [U, V, cyc_ite] = pre_V_cycle(F_U, F_V, v1, v2, ite_max, Vcycle_error, N, bottom, device)
K = log(N/bottom)/log(2);
U_cell = cell(1, K+1);
V_cell = cell(1, K+1);
F_U_cell = cell(1, K);
F_V_cell = cell(1, K);
F_U_cell{1} = F_U;
F_V_cell{1} = F_V;
if device == 'gpu'
    U_cell{1} = gpuArray.zeros(N+1, N);
    V_cell{1} = gpuArray.zeros(N, N+1);
else
    U_cell{1} = zeros(N+1, N);
    V_cell{1} = zeros(N, N+1);
end

r0 = sqrt(norm(F_U_cell{1}, 'fro')^2+norm(F_V_cell{1}, 'fro')^2);
cyc_ite = 0;
ts = 0;
for ite = 1:ite_max
    cyc_ite = cyc_ite+1;
    for i = 1:K+1
        if i ~= 1
            U = zeros(N+1, N);
            V = zeros(N, N+1);
        else
            U = U_cell{1};
            V = V_cell{1};

            F_U = F_U_cell{1};
            F_V = F_V_cell{1};
        end
        if i ~= K+1
            time_1 = toc;
            for t=1:v1
                [U, V] = sym_GS_iteration(U, V, F_U, F_V, N, device);
            end
            time_2 = toc;
            ts = ts + time_2 - time_1;
        else
            for ite__ = 1:v1+v2
                [U, V] = sym_GS_iteration(U, V, F_U, F_V, N, device);
            end
        end
        U_cell{i} = U;
        V_cell{i} = V;
        if i ~= 1 && i ~= K+1
            F_U_cell{i} = F_U;
            F_V_cell{i} = F_V;
        end
        if i ~= K+1
            [U, V] = apply_A(U, V, N, device);
            [F_U, F_V] = restrict_UV(F_U-U, F_V-V, N, device);
            N = N / 2;
        end
    end
    for i = K:-1:1
        [U, V] = lift_UV(U_cell{i+1}, V_cell{i+1}, N, device);
        N = N * 2;
        U = U_cell{i} + U;
        V = V_cell{i} + V;
        time_1 = toc;
        for t = 1:v2
            [U,V]=sym_GS_iteration(U,V,F_U_cell{i},F_V_cell{i},N,device);
        end
        time_2 = toc;
        ts = ts + time_2 - time_1;
        U_cell{i} = U;
        V_cell{i} = V;
    end
    [A_U, A_V] = apply_A(U, V, N, device);
    rh=sqrt(norm(A_U - F_U_cell{1}, 'fro')^2 + norm(A_V - F_V_cell{1}, 'fro')^2);
    if rh / r0 < Vcycle_error
        break
    end
end
end