N_array = [64, 128, 256, 512, 1024, 2048];

%使用设备为gpu时因为CUDA启动等原因，通常最开始的样例较慢，若需要测时间，推荐重复测试
for N = N_array
    fprintf("N = %d\n", N);
    h = 1/N;
    [F_U, F_V, U0, V0] = initialize_v_cycle(N);

    %==================参数设置================= 
    alpha = 1;
    device = 'cpu';%使用设备为'gpu'或'cpu'
    %==========================================

    tic;
    [U, V, P, Uzawa_ite_num, CG_iter_num] = Uzawa_iteration(F_U, F_V, N, alpha, device);
    toc

    if device ==  'gpu'
        U = gather(U);
        V = gather(V);
    end
    
    error = sqrt(norm(U(2:N, 1:N) - U0(2:N, 1:N), 'fro')^2 + norm(V(1:N, 2:N) - V0(1:N, 2:N), 'fro')^2);
    error = h * error;%计算误差
    fprintf("e_N = %g\n",error);
    fprintf("Uzawa迭代次数:%d\n",Uzawa_ite_num);
    fprintf("共轭梯度法迭代次数:")
    disp(CG_iter_num);
    clear;
end



