function [U, V ,Vcycle_ite_num] = V_cycle(F_U, F_V, v1, v2, N, bottom, tol, device)
K=log(N/bottom)/log(2);

F_P = zeros(N,N);
U_cell = cell(1,K+1);
V_cell = cell(1,K+1);
P_cell = cell(1,K+1);
F_U_cell = cell(1,K);
F_V_cell = cell(1,K);
F_P_cell = cell(1,K);

device_memory = device;
N_memory = N;

if device == 'gpu'
    U_cell{1} = gpuArray.zeros(N+1, N);
    V_cell{1} = gpuArray.zeros(N, N+1);
    P_cell{1} = gpuArray.zeros(N, N);
    F_U = gpuArray(F_U);
    F_V = gpuArray(F_V);
    F_P = gpuArray(F_P);
else
    U_cell{1} = zeros(N+1, N);
    V_cell{1} = zeros(N, N+1);
    P_cell{1} = zeros(N, N);
end


F_U_cell{1} = F_U;
F_V_cell{1} = F_V;
F_P_cell{1} = F_P;
r0 = sqrt(norm(F_U,'fro')^2 + norm(F_V,'fro')^2);
Vcycle_ite_num = 0;
while 1
    device = device_memory;
    Vcycle_ite_num = Vcycle_ite_num+1;
    for i = 1:K+1
        if i ~= 1 
            if (device_memory == 'gpu') & ((N_memory / 2^(i-1)) > tol)
                U = gpuArray.zeros(N+1, N);
                V = gpuArray.zeros(N, N+1);
                P = gpuArray.zeros(N, N);
            else
                device = 'cpu';
                U = zeros(N+1, N);
                V = zeros(N, N+1);
                P = zeros(N, N);
            end
        else
            U = U_cell{1};
            V = V_cell{1};
            P = P_cell{1};
            F_U = F_U_cell{1};
            F_V = F_V_cell{1};
            F_P = F_P_cell{1};
        end
        if i ~= K+1
            for k = 1:v1
                [U, V, P] = RBGS_iteration(U, V, P, F_U, F_V, F_P, N, device);
            end
        else
            for k = 1:2*bottom
                [U, V, P] = RBGS_iteration(U, V, P, F_U, F_V, F_P, N, device);
            end
        end

        U_cell{i} = U;
        V_cell{i} = V;
        P_cell{i} = P;
        if i ~= 1 && i ~= K+1
            F_U_cell{i} = F_U;
            F_V_cell{i} = F_V;
            F_P_cell{i} = F_P;
        end
        if i ~= K+1
            [A_U, A_V] = apply_A(U, V, N, device);
            [B_Pu, B_Pv] = apply_B(P, N, device);
            P = apply_Btrans(U, V, N);
            [F_U, F_V, F_P] = restrict(F_U-A_U-B_Pu, F_V-A_V-B_Pv, F_P-P,N, device);
            if device_memory == 'gpu' & ((N_memory / 2^(i-1)) < tol*1.5) & ((N_memory / 2^(i-1)) > tol*0.75)
                F_U = gather(F_U);
                F_V = gather(F_V);
                F_P = gather(F_P);
            end
            N=N/2;
        end
    end
    %N为底层的大小
    for i = K:-1:1
        if (device_memory == 'gpu') & ((N_memory / 2^(i-1)) > tol)
            device = device_memory;
        end
        [U, V, P] = lift(U_cell{i+1}, V_cell{i+1}, P_cell{i+1}, N, device);
        N = N * 2;
        U = U_cell{i} + U;
        V = V_cell{i} + V;
        P = P_cell{i} + P;
        for k = 1:v2
            [U, V, P] = RBGS_iteration(U, V, P, F_U_cell{i}, F_V_cell{i}, F_P_cell{i}, N, device);
        end
        U_cell{i} = U;
        V_cell{i} = V;
        P_cell{i} = P;
    end
    [A_U, A_V] = apply_A(U, V, N, device);
    [B_Pu, B_Pv] = apply_B(P, N, device);
    rh = sqrt(norm(A_U + B_Pu - F_U_cell{1}, 'fro')^2 + norm(A_V + B_Pv - F_V_cell{1}, 'fro')^2 + norm(apply_Btrans(U, V, N), 'fro')^2);
    if rh / r0 < 1e-8
        break
    end
end
fprintf("Vcycle迭代次数:%d\n", Vcycle_ite_num);
end
