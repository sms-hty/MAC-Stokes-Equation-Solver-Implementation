%计算B*P, 即P的梯度 Pu, Pv分别为u, v方向的梯度(u, v方向即x, y方向)
function [B_Pu, B_Pv] = apply_B(P, N, device)   
    kernel_vertical = [1; -1];  % 2x1 核，沿列方向差分
    kernel_horizontal = [1, -1];  % 1x2 核，沿行方向差分
    % 初始化输出矩阵
    if device == 'gpu'
        B_Pu = gpuArray.zeros(N+1, N); 
        B_Pv = gpuArray.zeros(N, N+1); 
        kernel_vertical = gpuArray(kernel_vertical);
        kernel_horizontal = gpuArray(kernel_horizontal);
    else
        B_Pu = zeros(N+1, N); 
        B_Pv = zeros(N, N+1); 
    end

    % 计算水平梯度 B_Pu（矩阵垂直方向差分）
    B_Pu(2:N, :) = conv2(P, kernel_vertical, 'valid') * N;
    
    % 计算垂直梯度 B_Pv（矩阵水平方向差分）
    B_Pv(:, 2:N) = conv2(P, kernel_horizontal, 'valid') * N;
end