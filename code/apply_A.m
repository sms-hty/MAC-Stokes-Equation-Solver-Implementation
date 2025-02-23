function [A_U, A_V] = apply_A(U, V, N, device)
%     h = 1/N;
%     
%     % A_U = A * U 
%     % size of U : (N+1)*N 
%     % u(i, j+1/2) <-> U(i, j)  
% 
% 
%     % 定义离散拉普拉斯卷积核
%     kernel = [0  1  0;
%               1 -4  1;
%               0  1  0];
% 
%     kernel_edge_left = [0 1;
%                       1 -3;
%                       0 1];
%     kernel_edge_right = [1 0;
%                       -3 1;
%                       1 0];
%     kernel_edge_low = [0 1 0;
%                       1 -3 1;];
%     kernel_edge_up = [1 -3 1;
%                        0 1 0];
%     scale = -1 / (h^2);
%     
%     % 处理 A_U
%     A_U = zeros(N+1, N);
%     % 内部区域 (i=2:N, j=2:N-1) 用卷积计算
%     A_U(2:N, 2:N-1) = scale * conv2(U, kernel, 'valid');
% 
%     % 上下边界
% %     for i = 2:N
% %         A_U(i,1) = -1/(h^2)*(U(i-1,1) + U(i+1,1) - 3*U(i,1) + U(i,2));
% %         A_U(i,N) = -1/(h^2)*(U(i-1,N) + U(i+1,N) - 3*U(i,N) + U(i,N-1));
% %     end
% %     A_U_new = scale * conv2(U(:, N-1:N), kernel_edge_up, 'valid');
% %     delta = A_U(2:N, N) - A_U_new;
% %     disp(max(abs(delta(:))))
% 
%     A_U(2:N, 1) = scale * conv2(U(:, 1:2), kernel_edge_left, 'valid');
%     A_U(2:N, N) = scale * conv2(U(:, N-1:N), kernel_edge_right, 'valid');
% 
%     % 左右边界
%     A_U(1,:) = U(1,:);
%     A_U(N+1,:) = U(N+1,:);
%     
%     % 处理 A_V（类似逻辑）
% 
%     % A_V = A * V 
%     % size of V : N*(N+1) 
%     % v(i+1/2, j) <-> V(i, j)  
% 
% 
% 
%     %内部区域
%     A_V = zeros(N, N+1);
%     A_V(2:N-1, 2:N) = scale * conv2(V, kernel, 'valid');
%     
%     % 边界处理
% %     for j = 2:N
% %         A_V(1,j) = -1/(h^2)*(V(2,j) - 3*V(1,j) + V(1,j+1) + V(1,j-1));
% %         A_V(N,j) = -1/(h^2)*(V(N-1,j) - 3*V(N,j) + V(N,j+1) + V(N,j-1));
% %     end
% %     A_V_new = scale * conv2(V(1:2, :), kernel_edge_low, 'valid');
% %     delta = A_V(1, 2:N) - A_V_new;
% %     disp(max(abs(delta(:))))
% 
%     A_V(1, 2:N) = scale * conv2(V(1:2, :), kernel_edge_low, 'valid');
%     A_V(N, 2:N) = scale * conv2(V(N-1:N, :), kernel_edge_up, 'valid');
% 
%     A_V(:,1) = V(:,1);
%     A_V(:,N+1) = V(:,N+1);

    % 将输入矩阵和卷积核转换为 gpuArray  
    h = 1/N;
    
    % 定义离散拉普拉斯卷积核
    kernel = [0  1  0;
              1 -4  1;
              0  1  0];

    
    kernel_edge_left = [0 1;
                        1 -3;
                        0 1];

    
    kernel_edge_right = [1 0;
                        -3 1;
                         1 0];

    
    kernel_edge_low = [0 1 0;
                       1 -3 1];

    
    kernel_edge_up = [1 -3 1;
                      0  1 0];

    
    scale = -N^2;

    if device == 'gpu'
%         U = gpuArray(U);
%         V = gpuArray(V);
        kernel = gpuArray(kernel); % 转换为 GPU 格式
        kernel_edge_left = gpuArray(kernel_edge_left);
        kernel_edge_right = gpuArray(kernel_edge_right);
        kernel_edge_low = gpuArray(kernel_edge_low);
        kernel_edge_up = gpuArray(kernel_edge_up);
        A_U = gpuArray.zeros(N+1, N);
        A_V = gpuArray.zeros(N, N+1);
    else
        A_U = zeros(N+1, N);
        A_V = zeros(N, N+1);
    end

    
    % 处理 A_U

    % 内部区域 (i=2:N, j=2:N-1) 用卷积计算
    A_U(2:N, 2:N-1) = scale * conv2(U, kernel, 'valid');
    
    % 上下边界
    A_U(2:N, 1) = scale * conv2(U(:, 1:2), kernel_edge_left, 'valid');
    A_U(2:N, N) = scale * conv2(U(:, N-1:N), kernel_edge_right, 'valid');
    
    % 左右边界
    A_U(1,:) = U(1,:);
    A_U(N+1,:) = U(N+1,:);
    
    % 处理 A_V（类似逻辑）

    % 内部区域
    A_V(2:N-1, 2:N) = scale * conv2(V, kernel, 'valid');
    
    % 边界处理
    A_V(1, 2:N) = scale * conv2(V(1:2, :), kernel_edge_low, 'valid');
    A_V(N, 2:N) = scale * conv2(V(N-1:N, :), kernel_edge_up, 'valid');
    
    A_V(:,1) = V(:,1);
    A_V(:,N+1) = V(:,N+1);
    
%     A_U = gather(A_U);
%     A_V = gather(A_V);
end