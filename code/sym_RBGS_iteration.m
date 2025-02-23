function [U_ite, V_ite] = sym_RBGS_iteration(U_ite, V_ite, F_U, F_V, N, device)
    h = 1/N;
    h_sq = h^2;
%     U_ite = U;
%     V_ite = V;
    % 生成红黑掩码
    if device == 'gpu'
%         [red_mask_U, black_mask_U] = create_red_mask_U(N);
%         [red_mask_V, black_mask_V] = create_red_mask_V(N);
%     
%         % ================== 红节点更新阶段 ==================
%         % 计算残差
%         [A_U, A_V] = apply_A(U, V, N, device);
%         residual_U = F_U - A_U;
%         residual_V = F_V - A_V;
%     
%         % 更新红色节点(i + j mod 2 == 1)
%         U(red_mask_U) = U(red_mask_U) + (h_sq / 4) * residual_U(red_mask_U);
%         V(red_mask_V) = V(red_mask_V) + (h_sq / 4) * residual_V(red_mask_V);
%         
%         %边界处理
%         U(2:2:end, 1) = U(2:2:end, 1) + (h_sq / 3) * residual_U(2:2:end, 1);
%         U(end-2:-2:1, N) = U(end-2:-2:1, N) + (h_sq / 3) * residual_U(end-2:-2:1, N);
%         V(1, 2:2:end) = V(1, 2:2:end) + (h_sq / 3) * residual_V(1, 2:2:end);
%         V(N, end-2:-2:1) = V(N, end-2:-2:1) + (h_sq / 3) * residual_V(N, end-2:-2:1);   
%     
%         % ================== 黑节点更新阶段 ==================
%         % 重新计算残差（使用更新后的红节点）
%         [A_U, A_V] = apply_A(U, V, N, device);
%         residual_U = F_U - A_U;
%         residual_V = F_V - A_V;
%     
%         % 更新黑色节点(i + j mod 2 == 0)
%         U(black_mask_U) = U(black_mask_U) + (h_sq / 4) * residual_U(black_mask_U);
%         V(black_mask_V) = V(black_mask_V) + (h_sq / 4) * residual_V(black_mask_V);
%         
%         %边界处理
%         U(1:2:end-2, 1) = U(1:2:end-2, 1) + (h_sq / 3) * residual_U(1:2:end-2, 1);
%         U(N:-2:1, N) = U(N:-2:1, N) + (h_sq / 3) * residual_U(N:-2:1, N);
%         V(1, 1:2:end-2) = V(1, 1:2:end-2) + (h_sq / 3) * residual_V(1, 1:2:end-2);
%         V(end, N:-2:1) = V(end, N:-2:1) + (h_sq / 3) * residual_V(end, N:-2:1); 
%     
%         % ================== 反向更新(先黑后红)====================
%         [A_U, A_V] = apply_A(U, V, N, device);
%         residual_U = F_U - A_U;
%         residual_V = F_V - A_V;
%     
%         % 更新黑色节点(i + j mod 2 == 0)
%         U(black_mask_U) = U(black_mask_U) + (h_sq / 4) * residual_U(black_mask_U);
%         V(black_mask_V) = V(black_mask_V) + (h_sq / 4) * residual_V(black_mask_V);
%         
%         %边界处理
%         U(1:2:end-2, 1) = U(1:2:end-2, 1) + (h_sq / 3) * residual_U(1:2:end-2, 1);
%         U(N:-2:1, N) = U(N:-2:1, N) + (h_sq / 3) * residual_U(N:-2:1, N);
%         V(1, 1:2:end-2) = V(1, 1:2:end-2) + (h_sq / 3) * residual_V(1, 1:2:end-2);
%         V(end, N:-2:1) = V(end, N:-2:1) + (h_sq / 3) * residual_V(end, N:-2:1); 
%        
%         % 计算残差
%         [A_U, A_V] = apply_A(U, V, N, device);
%         residual_U = F_U - A_U;
%         residual_V = F_V - A_V;
%     
%         % 更新红色节点(i + j mod 2 == 1)
%         U(red_mask_U) = U(red_mask_U) + (h_sq / 4) * residual_U(red_mask_U);
%         V(red_mask_V) = V(red_mask_V) + (h_sq / 4) * residual_V(red_mask_V);
%         
%         %边界处理
%         U(2:2:end, 1) = U(2:2:end, 1) + (h_sq / 3) * residual_U(2:2:end, 1);
%         U(end-2:-2:1, N) = U(end-2:-2:1, N) + (h_sq / 3) * residual_U(end-2:-2:1, N);
%         V(1, 2:2:end) = V(1, 2:2:end) + (h_sq / 3) * residual_V(1, 2:2:end);
%         V(N, end-2:-2:1) = V(N, end-2:-2:1) + (h_sq / 3) * residual_V(N, end-2:-2:1);  
        N = gpuArray(N);
        h = 1/N;
        h_sq = h^2;
%         omega = 1;%松弛因子
%         h_sq = h_sq;
        % 生成红黑掩码
        [red_mask_U, black_mask_U] = create_red_mask_U(N);
        [red_mask_V, black_mask_V] = create_red_mask_V(N);
    
        % ================== 红节点更新阶段 ==================
        % 计算残差
        [A_U, A_V] = apply_A(U_ite, V_ite, N, device);
        residual_U = F_U - A_U ;
        residual_V = F_V - A_V;
        
        h_temp = h_sq / 4;
        % 更新红色节点(i + j mod 2 == 1)
        U_ite(red_mask_U) = U_ite(red_mask_U) + h_temp * residual_U(red_mask_U);
        V_ite(red_mask_V) = V_ite(red_mask_V) + h_temp * residual_V(red_mask_V);
        
        h_temp = h_sq / 3;
        %边界处理
        U_ite(2:2:end, 1) = U_ite(2:2:end, 1) + h_temp * residual_U(2:2:end, 1);
        U_ite(end-2:-2:3, N) = U_ite(end-2:-2:3, N) + h_temp * residual_U(end-2:-2:3, N);
        V_ite(1, 2:2:end) = V_ite(1, 2:2:end) + h_temp * residual_V(1, 2:2:end);
        V_ite(N, end-2:-2:3) = V_ite(N, end-2:-2:3) + h_temp * residual_V(N, end-2:-2:3);   
    
    %     U_ite([1, N+1], 1:N) = U([1, N+1], 1:N);
    %     V_ite(1:N, [1, N+1]) = V(1:N, [1, N+1]);
    
        % ================== 黑节点更新阶段 ==================
        % 重新计算残差（使用更新后的红节点）
        [A_U, A_V] = apply_A(U_ite, V_ite, N, device);

        residual_U = F_U - A_U;
        residual_V = F_V - A_V;
        
        h_temp = h_sq / 4;
        % 更新黑色节点(i + j mod 2 == 0)
        U_ite(black_mask_U) = U_ite(black_mask_U) + h_temp * residual_U(black_mask_U);
        V_ite(black_mask_V) = V_ite(black_mask_V) + h_temp * residual_V(black_mask_V);
        
        h_temp = h_sq / 3;
        %边界处理
        U_ite(1:2:end-2, 1) = U_ite(1:2:end-2, 1) + h_temp * residual_U(1:2:end-2, 1);
        U_ite(N:-2:1, N) = U_ite(N:-2:1, N) + h_temp * residual_U(N:-2:1, N);
        V_ite(1, 1:2:end-2) = V_ite(1, 1:2:end-2) + h_temp * residual_V(1, 1:2:end-2);
        V_ite(end, N:-2:1) = V_ite(end, N:-2:1) + h_temp * residual_V(end, N:-2:1);  
        
        %反向更新
        % ================== 黑节点更新阶段 ==================
        % 重新计算残差（使用更新后的红节点）
        [A_U, A_V] = apply_A(U_ite, V_ite, N, device);

        residual_U = F_U - A_U;
        residual_V = F_V - A_V;
        
        h_temp = h_sq / 4;
        % 更新黑色节点(i + j mod 2 == 0)
        U_ite(black_mask_U) = U_ite(black_mask_U) + h_temp * residual_U(black_mask_U);
        V_ite(black_mask_V) = V_ite(black_mask_V) + h_temp * residual_V(black_mask_V);
        
        h_temp = h_sq / 3;
        %边界处理
        U_ite(1:2:end-2, 1) = U_ite(1:2:end-2, 1) + h_temp * residual_U(1:2:end-2, 1);
        U_ite(N:-2:1, N) = U_ite(N:-2:1, N) + h_temp * residual_U(N:-2:1, N);
        V_ite(1, 1:2:end-2) = V_ite(1, 1:2:end-2) + h_temp * residual_V(1, 1:2:end-2);
        V_ite(end, N:-2:1) = V_ite(end, N:-2:1) + h_temp * residual_V(end, N:-2:1);  

        % ================== 红节点更新阶段 ==================
        % 计算残差
        [A_U, A_V] = apply_A(U_ite, V_ite, N, device);
        residual_U = F_U - A_U;
        residual_V = F_V - A_V;
        
        h_temp = h_sq / 4;
        % 更新红色节点(i + j mod 2 == 1)
        U_ite(red_mask_U) = U_ite(red_mask_U) + h_temp * residual_U(red_mask_U);
        V_ite(red_mask_V) = V_ite(red_mask_V) + h_temp * residual_V(red_mask_V);
        
        h_temp = h_sq / 3;
        %边界处理
        U_ite(2:2:end, 1) = U_ite(2:2:end, 1) + h_temp * residual_U(2:2:end, 1);
        U_ite(end-2:-2:3, N) = U_ite(end-2:-2:3, N) + h_temp * residual_U(end-2:-2:3, N);
        V_ite(1, 2:2:end) = V_ite(1, 2:2:end) + h_temp * residual_V(1, 2:2:end);
        V_ite(N, end-2:-2:3) = V_ite(N, end-2:-2:3) + h_temp * residual_V(N, end-2:-2:3); 



        %cpu
    else
        h = 1/N;
        h_sq = h^2;
        h_sq_1 = 1/h^2;
    %     omega = 1; % 松弛因子
        % ================== 红节点更新阶段 ==================
        % 更新红色节点(i + j mod 2 == 1)
        h_temp = h_sq / 4;
        for i = 2:N
            mod_j = mod(i+1, 2) + 2;
            for j = mod_j:2:N-1
                res = h_temp * (F_U(i, j) + h_sq_1*(U_ite(i,j+1)+U_ite(i,j-1)+U_ite(i-1,j)+U_ite(i+1,j)-4*U_ite(i,j)));
                U_ite(i, j) = U_ite(i, j) + res;
            end
        end
    
        for i = 2:N-1
            mod_j = mod(i+1, 2) + 2;
            for j = mod_j:2:N
                res = h_temp * (F_V(i, j) + h_sq_1*(V_ite(i,j+1)+V_ite(i,j-1)+V_ite(i-1,j)+V_ite(i+1,j)-4*V_ite(i,j)));
                V_ite(i, j) = V_ite(i, j) + res;
            end
        end
        
        h_temp = h_sq / 3;        
        
        %边界处理
        for i = 2:2:N
            res = h_temp * (F_U(i, 1) + h_sq_1*(U_ite(i, 2)+U_ite(i-1, 1)+U_ite(i+1, 1)-3*U_ite(i, 1)));
            U_ite(i, 1) = U_ite(i, 1) + res;
    
            res = h_temp * (F_V(1, i) + h_sq_1*(V_ite(1 ,i+1)+V_ite(1, i-1)+V_ite(2, i)-3*V_ite(1, i)));
            V_ite(1, i) = V_ite(1, i) + res; 
        end
    
        for i = 3:2:N-1
            res = h_temp * (F_U(i, N) + h_sq_1*(U_ite(i, N-1)+U_ite(i-1, N)+U_ite(i+1, N)-3*U_ite(i, N)));
            U_ite(i, N) = U_ite(i, N) + res;
    
            res = h_temp * (F_V(N, i) + h_sq_1*(V_ite(N ,i+1)+V_ite(N, i-1)+V_ite(N-1, i)-3*V_ite(N, i)));
            V_ite(N, i) = V_ite(N, i) + res; 
        end
    
        % ================== 黑节点更新阶段 ==================
        % 重新计算残差（使用更新后的红节点）
        h_temp = h_sq / 4;
        for i = 2:N
            mod_j = mod(i, 2) + 2;
            for j = mod_j:2:N-1
                res = h_temp * (F_U(i, j) + h_sq_1*(U_ite(i,j+1)+U_ite(i,j-1)+U_ite(i-1,j)+U_ite(i+1,j)-4*U_ite(i,j)));
                U_ite(i, j) = U_ite(i, j) + res;
            end
        end
    
        for i = 2:N-1
            mod_j = mod(i, 2) + 2;
            for j = mod_j:2:N
                res = h_temp * (F_V(i, j) + h_sq_1*(V_ite(i,j+1)+V_ite(i,j-1)+V_ite(i-1,j)+V_ite(i+1,j)-4*V_ite(i,j)));
                V_ite(i, j) = V_ite(i, j) + res;
            end
        end
        
        h_temp = h_sq / 3;
        %边界处理
        for i = 3:2:N
            res = h_temp * (F_U(i, 1) + h_sq_1*(U_ite(i, 2)+U_ite(i-1, 1)+U_ite(i+1, 1)-3*U_ite(i, 1)));
            U_ite(i, 1) = U_ite(i, 1) + res;
    
            res = h_temp * (F_V(1, i) + h_sq_1*(V_ite(1 ,i+1)+V_ite(1, i-1)+V_ite(2, i)-3*V_ite(1, i)));
            V_ite(1, i) = V_ite(1, i) + res; 
        end
    
        for i = 2:2:N
            res = h_temp * (F_U(i, N) + h_sq_1*(U_ite(i, N-1)+U_ite(i-1, N)+U_ite(i+1, N)-3*U_ite(i, N)));
            U_ite(i, N) = U_ite(i, N) + res;
    
            res = h_temp * (F_V(N, i) + h_sq_1*(V_ite(N ,i+1)+V_ite(N, i-1)+V_ite(N-1, i)-3*V_ite(N, i)));
            V_ite(N, i) = V_ite(N, i) + res; 
        end

        %反向更新

        % ================== 黑节点更新阶段 ==================
        % 重新计算残差（使用更新后的红节点）
        h_temp = h_sq / 4;
        for i = 2:N
            mod_j = mod(i, 2) + 2;
            for j = mod_j:2:N-1
                res = h_temp * (F_U(i, j) + h_sq_1*(U_ite(i,j+1)+U_ite(i,j-1)+U_ite(i-1,j)+U_ite(i+1,j)-4*U_ite(i,j)));
                U_ite(i, j) = U_ite(i, j) + res;
            end
        end
    
        for i = 2:N-1
            mod_j = mod(i, 2) + 2;
            for j = mod_j:2:N
                res = h_temp * (F_V(i, j) + h_sq_1*(V_ite(i,j+1)+V_ite(i,j-1)+V_ite(i-1,j)+V_ite(i+1,j)-4*V_ite(i,j)));
                V_ite(i, j) = V_ite(i, j) + res;
            end
        end
        
        h_temp = h_sq / 3;
        %边界处理
        for i = 3:2:N
            res = h_temp * (F_U(i, 1) + h_sq_1*(U_ite(i, 2)+U_ite(i-1, 1)+U_ite(i+1, 1)-3*U_ite(i, 1)));
            U_ite(i, 1) = U_ite(i, 1) + res;
    
            res = h_temp * (F_V(1, i) + h_sq_1*(V_ite(1 ,i+1)+V_ite(1, i-1)+V_ite(2, i)-3*V_ite(1, i)));
            V_ite(1, i) = V_ite(1, i) + res; 
        end
    
        for i = 2:2:N
            res = h_temp * (F_U(i, N) + h_sq_1*(U_ite(i, N-1)+U_ite(i-1, N)+U_ite(i+1, N)-3*U_ite(i, N)));
            U_ite(i, N) = U_ite(i, N) + res;
    
            res = h_temp * (F_V(N, i) + h_sq_1*(V_ite(N ,i+1)+V_ite(N, i-1)+V_ite(N-1, i)-3*V_ite(N, i)));
            V_ite(N, i) = V_ite(N, i) + res; 
        end
        % ================== 红节点更新阶段 ==================
        % 更新红色节点(i + j mod 2 == 1)
        h_temp = h_sq / 4;
        for i = 2:N
            mod_j = mod(i+1, 2) + 2;
            for j = mod_j:2:N-1
                res = h_temp * (F_U(i, j) + h_sq_1*(U_ite(i,j+1)+U_ite(i,j-1)+U_ite(i-1,j)+U_ite(i+1,j)-4*U_ite(i,j)));
                U_ite(i, j) = U_ite(i, j) + res;
            end
        end
    
        for i = 2:N-1
            mod_j = mod(i+1, 2) + 2;
            for j = mod_j:2:N
                res = h_temp * (F_V(i, j) + h_sq_1*(V_ite(i,j+1)+V_ite(i,j-1)+V_ite(i-1,j)+V_ite(i+1,j)-4*V_ite(i,j)));
                V_ite(i, j) = V_ite(i, j) + res;
            end
        end
        
        h_temp = h_sq / 3;        
        
        %边界处理
        for i = 2:2:N
            res = h_temp * (F_U(i, 1) + h_sq_1*(U_ite(i, 2)+U_ite(i-1, 1)+U_ite(i+1, 1)-3*U_ite(i, 1)));
            U_ite(i, 1) = U_ite(i, 1) + res;
    
            res = h_temp * (F_V(1, i) + h_sq_1*(V_ite(1 ,i+1)+V_ite(1, i-1)+V_ite(2, i)-3*V_ite(1, i)));
            V_ite(1, i) = V_ite(1, i) + res; 
        end
    
        for i = 3:2:N-1
            res = h_temp * (F_U(i, N) + h_sq_1*(U_ite(i, N-1)+U_ite(i-1, N)+U_ite(i+1, N)-3*U_ite(i, N)));
            U_ite(i, N) = U_ite(i, N) + res;
    
            res = h_temp * (F_V(N, i) + h_sq_1*(V_ite(N ,i+1)+V_ite(N, i-1)+V_ite(N-1, i)-3*V_ite(N, i)));
            V_ite(N, i) = V_ite(N, i) + res; 
        end
    end
end

function [red_mask, black_mask] = create_red_mask_U(N)
    [I, J] = ndgrid(1:N+1, 1:N);
    I = gpuArray(I);
    J = gpuArray(J);
    red_mask = mod(I + J, 2) == 1;
    red_mask([1, N+1], :) = false;        
    red_mask(:, [1, N]) = false;


    black_mask = ~red_mask;
    black_mask([1, N+1], :) = false;     
    black_mask(:, [1, N]) = false;
end

% 辅助函数：创建速度场V的红黑掩码（Nx(N+1)网格）
function [red_mask, black_mask] = create_red_mask_V(N)
    [I, J] = ndgrid(1:N, 1:N+1);
    I = gpuArray(I);
    J = gpuArray(J);
    red_mask = mod(I + J, 2) == 1;
    red_mask([1, N], :) = false;        
    red_mask(:, [1, N+1]) = false;

    black_mask = ~red_mask;
    black_mask([1, N], :) = false;     
    black_mask(:, [1, N+1]) = false;
end