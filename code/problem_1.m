N_array = [64, 128, 256, 512, 1024, 2048, 4096];
% N_array = [64, 64, 64, 64, 64, 64];
% N_array = [128, 128, 128, 128, 128, 128];
% N_array = [256, 256, 256, 256, 256, 256];
% N_array = [512, 512, 512, 512, 512, 512];
% N_array = [1024, 1024, 1024, 1024, 1024, 1024];
% N_array = [2048, 2048, 2048, 2048, 2048, 2048];
% N_array = [4096, 4096, 4096, 4096];

%使用设备为gpu时因为CUDA启动等原因，通常最开始的样例较慢，若需要测时间，推荐重复测试
for N = N_array

    %=====================参数设置====================
    device = 'gpu'; %使用设备为'gpu'或'cpu'
    bottom = 4;%最低层尺寸
    v1 = 1;%\nu_1
    v2 = 1;%\nu_2
    %================================================

    fprintf("\n")
    fprintf("N = %d\n", N);
    h = 1/N;
    tol = min(N-4, 1000);
    if N == 4096
        tol = 2000;
    end
    [F_U, F_V, U0, V0] = initialize_v_cycle(N);%初始化

    tic
    [U, V, Vcycle_ite_num] = V_cycle(F_U, F_V, v1, v2, N, bottom, tol, device);%V-cycle求解
    toc

    if device == 'gpu'
        U = gather(U);
        V = gather(V);
    end
    error = sqrt(norm(U(2:N, 1:N) - U0(2:N, 1:N), 'fro')^2 + norm(V(1:N, 2:N) - V0(1:N, 2:N), 'fro')^2);
    error = h * error;%计算误差
    fprintf("e_N = %g\n\n", error);
    clear;
end



