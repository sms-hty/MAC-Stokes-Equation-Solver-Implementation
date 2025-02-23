N_array = [64, 128, 256, 512, 1024, 2048, 4096];

for N = N_array
    fprintf("N = %d\n", N);
    h = 1/N;
    [F_U, F_V, U0, V0]=initialize_v_cycle(N);%初始化，计算F和真解

    %====================参数设置=======================
    alpha = 1;
    v1 = 2;
    v2 = 2;
    bottom = 2;%底层尺寸
    PCG_ite_max = 10;%PCG最大迭代次数
    PCG_epsilon = 1e-6;%PCG过程中的残差限制
    tau = 1e-3;
    Vcycle_ite_max = 3;%Vcycle最大迭代次数
    Vcycle_error = 1e-6; 
    %===================================================

    device = 'cpu';%除非将顺序对称GS迭代替换为了红黑对称GS迭代，最好不要改为'gpu'
    tic;
    [U,V,P,Uzawa_ite_num] = Inexact_Uzawa_iteration(F_U, F_V, N, alpha, v1, v2, PCG_ite_max, Vcycle_ite_max, Vcycle_error, tau, bottom, PCG_epsilon, device);
    if device == 'gpu'
        U = gather(U);
        V = gather(V);
    end
    toc
    error = sqrt(norm(U(2:N, 1:N) - U0(2:N, 1:N), 'fro')^2 + norm(V(1:N, 2:N) - V0(1:N, 2:N), 'fro')^2);
    error = h * error;%计算误差
    fprintf("e_N = %g\n\n", error);
end




