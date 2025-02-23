% 计算B^T * U，其中U由两个方向速度分量(U, V)组成，返回P
function P = apply_Btrans(U, V, N)
    kernel_U = [1; -1];  % 1x2 核，沿列方向差分
    kernel_V = [1, -1];  % 2x1 核，沿行方向差分

    
    
    % 计算水平散度（U 的差分）
    % 输入 U 尺寸: (N+1) x N，卷积后得到 N x N
    div_U = conv2(U, kernel_U, 'valid') * N;
    
    % 计算垂直散度（V 的差分）
    % 输入 V 尺寸: N x (N+1)，卷积后得到 N x N
    div_V = conv2(V, kernel_V, 'valid') * N;
    
    % 合并散度场
    P = -(div_U + div_V);
end