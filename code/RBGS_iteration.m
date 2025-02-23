%[A  B  * [U  = [F
%B^T 0]    V]    D]
function [U_ite, V_ite, P_ite] = RBGS_iteration(U_ite, V_ite, P_ite, F_U, F_V, D, N, device)
    if device == 'gpu'
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
        [B_Pu, B_Pv] = apply_B(P_ite, N, device);
        residual_U = F_U - (A_U + B_Pu);
        residual_V = F_V - (A_V + B_Pv);
        
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

        residual_U = F_U - (A_U + B_Pu);
        residual_V = F_V - (A_V + B_Pv);
        
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
      


        [red_mask_P, black_mask_P] = create_red_mask_lattice(N);
        Residual_P = apply_Btrans(U_ite, V_ite, N) - D;

    % %     更新红色内部格子:(i,j),(i+1,j),(i,j+1),(i+1,j+1).i+j mod2 ==1
    % %     2<=i,j<=N-1
        
        kernel_P = gpuArray([0, -0.25, 0; -0.25, 1, -0.25; 0, -0.25, 0]);
        kernel_U = gpuArray([-1; 1]);
        kernel_V = gpuArray([-1, 1]);
        %tic
        %红格子
        %(2:N-1, 2:N-1)红格+(1, 2:2:N-1)+(N, 3:2:N-1)+(2:2:N-1, 1)+(3:2:N-1, N)+(N, 1)+(1, N)
        
        if N > 2
            h_temp = h / 4;
            %更新P--red-inner
            Res_P_temp = Residual_P(2:N-1, 2:N-1).*red_mask_P(2:N-1, 2:N-1);
            P_ite = P_ite + conv2(Res_P_temp, kernel_P, 'full');
        
            %更新U--red-inner
            U_ite(2:N, 2:N-1) = U_ite(2:N, 2:N-1) + h_temp * conv2(Res_P_temp, kernel_U, 'full');
            
            %更新V--red-inner
            V_ite(2:N-1, 2:N) = V_ite(2:N-1, 2:N) + h_temp * conv2(Res_P_temp, kernel_V, 'full');
        end
        %更新边界
        h_temp = h / 3;
        %(1, 2:2:N-1)->U = 0
        temp = Residual_P(1, 2:2:N-1);
        P_ite(1, 2:2:N-1) = P_ite(1, 2:2:N-1) + temp;
        P_ite(1, 1:2:N-2) = P_ite(1, 1:2:N-2) - 1/3 * temp;
        P_ite(1, 3:2:N) = P_ite(1, 3:2:N) - 1/3 * temp;
        P_ite(2, 2:2:N-1) = P_ite(2, 2:2:N-1) - 1/3 * temp;
        
        U_ite(2, 2:2:N-1) = U_ite(2, 2:2:N-1) + h_temp * temp;
        V_ite(1, 2:2:N-1) = V_ite(1, 2:2:N-1) - h_temp * temp;
        V_ite(1, 3:2:N) = V_ite(1, 3:2:N) + h_temp * temp;
    
        %(N, 3:2:N-1)->U = 0
        temp = Residual_P(N, 3:2:N-1);
        P_ite(N, 3:2:N-1) = P_ite(N, 3:2:N-1) + temp;
        P_ite(N, 2:2:N-2) = P_ite(N, 2:2:N-2) - 1/3 * temp;
        P_ite(N, 4:2:N) = P_ite(N, 4:2:N) - 1/3 * temp;
        P_ite(N-1, 3:2:N-1) = P_ite(N-1, 3:2:N-1) - 1/3 * temp;
    
        U_ite(N, 3:2:N-1) = U_ite(N, 3:2:N-1) - h_temp * temp;
        V_ite(N, 3:2:N-1) = V_ite(N, 3:2:N-1) - h_temp * temp;
        V_ite(N, 4:2:N) = V_ite(N, 4:2:N) + h_temp * temp;
    
        %(2:2:N-1, 1)->V = 0
        temp = Residual_P(2:2:N-1, 1);
        P_ite(2:2:N-1, 1) = P_ite(2:2:N-1, 1) + temp;
        P_ite(1:2:N-2, 1) = P_ite(1:2:N-2, 1) - 1/3 * temp;
        P_ite(3:2:N, 1) = P_ite(3:2:N, 1) - 1/3 * temp;
        P_ite(2:2:N-1, 2) = P_ite(2:2:N-1, 2) - 1/3 * temp; 
        
        U_ite(2:2:N-1, 1) = U_ite(2:2:N-1, 1) - h_temp * temp;
        U_ite(3:2:N, 1) = U_ite(3:2:N, 1) + h_temp * temp;
        V_ite(2:2:N-1, 2) = V_ite(2:2:N-1, 2) + h_temp * temp;
    
        %(3:2:N-1, N)->V = 0
        temp = Residual_P(3:2:N-1, N);
        P_ite(3:2:N-1, N) = P_ite(3:2:N-1, N) + temp;
        P_ite(2:2:N-2, N) = P_ite(2:2:N-2, N) - 1/3 * temp;
        P_ite(4:2:N, N) = P_ite(4:2:N, N) - 1/3 * temp;
        P_ite(3:2:N-1, N-1) = P_ite(3:2:N-1, N-1) - 1/3 * temp;
    
        U_ite(3:2:N-1, N) = U_ite(3:2:N-1, N) - h_temp * temp;
        U_ite(4:2:N, N) = U_ite(4:2:N, N) + h_temp * temp;
        V_ite(3:2:N-1, N) = V_ite(3:2:N-1, N) - h_temp * temp;   
        
        h_temp = h /2;
        %(N, 1)
        temp = Residual_P(N, 1);
        P_ite(N, 1) = P_ite(N, 1) + temp;
        P_ite(N, 2) = P_ite(N, 2) - 1/2 * temp;
        P_ite(N-1, 1) = P_ite(N-1, 1) - 1/2 * temp;
    
        U_ite(N, 1) = U_ite(N, 1) - h_temp * temp;
        V_ite(N, 2) = V_ite(N, 2) + h_temp * temp;
    
        %(1, N)
        temp = Residual_P(1, N);
        P_ite(1, N) = P_ite(1, N) + temp;
        P_ite(2, N) = P_ite(2, N) - 1/2 * temp;
        P_ite(1, N-1) = P_ite(1, N-1) - 1/2 * temp;
    
        U_ite(2, N) = U_ite(2, N) + h_temp * temp;
        V_ite(1, N) = V_ite(1, N) - h_temp * temp;
    
    
        Residual_P = apply_Btrans(U_ite, V_ite, N) - D;
        %黑格子
        %(2:N-1, 2:N-1)黑格+(1, 3:2:N-1)+(N, 2:2:N-2)+(3:2:N-1, 1)+(2:2:N-2, N)
        %+(1, 1)+(N, N)
    
        %更新P--balck-inner
        %S = conv2(Residual_P(2:N-1, 2:N-1).*black_mask_P(2:N-1, 2:N-1), kernel_P, 'full');
        if N > 2
            h_temp = h / 4;
            Res_P_temp = Residual_P(2:N-1, 2:N-1).*black_mask_P(2:N-1, 2:N-1);
            P_ite = P_ite + conv2(Res_P_temp, kernel_P, 'full');
        
            %更新U--black-inner
            %S = conv2(Residual_P(2:N-1, 2:N-1).*black_mask_P(2:N-1, 2:N-1), kernel_U, 'full');
            U_ite(2:N, 2:N-1) = U_ite(2:N, 2:N-1) + h_temp * conv2(Res_P_temp, kernel_U, 'full');
            
            %更新V--black-inner
            %S = conv2(Residual_P(2:N-1, 2:N-1).*black_mask_P(2:N-1, 2:N-1), kernel_V, 'full');
            V_ite(2:N-1, 2:N) = V_ite(2:N-1, 2:N) + h_temp * conv2(Res_P_temp, kernel_V, 'full');
        end
        h_temp = h / 3;
        %(1, 3:2:N-1)->U = 0
        temp = Residual_P(1, 3:2:N-1);
        P_ite(1, 3:2:N-1) = P_ite(1, 3:2:N-1) + temp;
        P_ite(1, 2:2:N-2) = P_ite(1, 2:2:N-2) - 1/3 * temp;
        P_ite(1, 4:2:N) = P_ite(1, 4:2:N) - 1/3 * temp;
        P_ite(2, 3:2:N-1) = P_ite(2, 3:2:N-1) - 1/3 * temp;
        
        U_ite(2, 3:2:N-1) = U_ite(2, 3:2:N-1) + h_temp * temp;
        V_ite(1, 3:2:N-1) = V_ite(1, 3:2:N-1) - h_temp * temp;
        V_ite(1, 4:2:N) = V_ite(1, 4:2:N) + h_temp * temp;
    
        %(N, 2:2:N-2)->U = 0
        temp = Residual_P(N, 2:2:N-2);
        P_ite(N, 2:2:N-2) = P_ite(N, 2:2:N-2) + temp;
        P_ite(N, 1:2:N-3) = P_ite(N, 1:2:N-3) - 1/3 * temp;
        P_ite(N, 3:2:N-1) = P_ite(N, 3:2:N-1) - 1/3 * temp;
        P_ite(N-1, 2:2:N-2) = P_ite(N-1, 2:2:N-2) - 1/3 * temp;
    
        U_ite(N, 2:2:N-2) = U_ite(N, 2:2:N-2) - h_temp * temp;
        V_ite(N, 2:2:N-2) = V_ite(N, 2:2:N-2) - h_temp * temp;
        V_ite(N, 3:2:N-1) = V_ite(N, 3:2:N-1) + h_temp * temp;
    
        %(3:2:N-1, 1)->V = 0
        temp = Residual_P(3:2:N-1, 1);
        P_ite(3:2:N-1, 1) = P_ite(3:2:N-1, 1) + temp;
        P_ite(2:2:N-2, 1) = P_ite(2:2:N-2, 1) - 1/3 * temp;
        P_ite(4:2:N, 1) = P_ite(4:2:N, 1) - 1/3 * temp;
        P_ite(3:2:N-1, 2) = P_ite(3:2:N-1, 2) - 1/3 * temp;
    
        U_ite(3:2:N-1, 1) = U_ite(3:2:N-1, 1) - h_temp * temp;
        U_ite(4:2:N, 1) = U_ite(4:2:N, 1) + h_temp * temp;
        V_ite(3:2:N-1, 2) = V_ite(3:2:N-1, 2) + h_temp * temp;
    
        %(2:2:N-2, N)->V = 0
        temp = Residual_P(2:2:N-2, N);
        P_ite(2:2:N-2, N) = P_ite(2:2:N-2, N) + temp;
        P_ite(1:2:N-3, N) = P_ite(1:2:N-3, N) - 1/3 * temp;
        P_ite(3:2:N-1, N) = P_ite(3:2:N-1, N) - 1/3 * temp;
        P_ite(2:2:N-2, N-1) = P_ite(2:2:N-2, N-1) - 1/3 * temp;
    
        U_ite(2:2:N-2, N) = U_ite(2:2:N-2, N) - h_temp * temp;
        U_ite(3:2:N-1, N) = U_ite(3:2:N-1, N) + h_temp * temp;
        V_ite(2:2:N-2, N) = V_ite(2:2:N-2, N) - h_temp * temp;
    
        h_temp = h / 2;
        %(1, 1)
        temp = Residual_P(1, 1);
        P_ite(1, 1) = P_ite(1, 1) + temp;
        P_ite(2, 1) = P_ite(2, 1) - 1/2 * temp;
        P_ite(1, 2) = P_ite(1, 2) - 1/2 * temp;
        
        U_ite(2, 1) = U_ite(2, 1) + h_temp * temp;
        V_ite(1, 2) = V_ite(1, 2) + h_temp * temp;
    
        %(N, N)
        temp = Residual_P(N, N);
        P_ite(N, N) = P_ite(N, N) + temp;
        P_ite(N-1, N) = P_ite(N-1, N) - 1/2 * temp;
        P_ite(N, N-1) = P_ite(N, N-1) - 1/2 * temp;
    
        U_ite(N, N) = U_ite(N, N) - h_temp * temp;
        V_ite(N, N) = V_ite(N, N) - h_temp * temp;

    else
        h = 1/N;
        h_sq = h^2;
        h_sq_1 = 1/h^2;
    %     omega = 1; % 松弛因子
        % ================== 红节点更新阶段 ==================
        % 更新红色节点(i + j mod 2 == 1)h_
        h_temp = h_sq / 4;
        for i = 2:N
            mod_j = mod(i+1, 2) + 2;
            for j = mod_j:2:N-1
                res = h_temp * (F_U(i, j) + h_sq_1*(U_ite(i,j+1)+U_ite(i,j-1)+U_ite(i-1,j)+U_ite(i+1,j)-4*U_ite(i,j))...
                    -N*(P_ite(i,j)-P_ite(i-1,j)));
                U_ite(i, j) = U_ite(i, j) + res;
            end
        end
    
        for i = 2:N-1
            mod_j = mod(i+1, 2) + 2;
            for j = mod_j:2:N
                res = h_temp * (F_V(i, j) + h_sq_1*(V_ite(i,j+1)+V_ite(i,j-1)+V_ite(i-1,j)+V_ite(i+1,j)-4*V_ite(i,j))...
                    -N*(P_ite(i,j)-P_ite(i,j-1)));
                V_ite(i, j) = V_ite(i, j) + res;
            end
        end
        
        h_temp = h_sq / 3;        
        
        %边界处理
        for i = 2:2:N
            res = h_temp * (F_U(i, 1) + h_sq_1*(U_ite(i, 2)+U_ite(i-1, 1)+U_ite(i+1, 1)-3*U_ite(i, 1))...
                    -N*(P_ite(i, 1)-P_ite(i-1, 1)));
            U_ite(i, 1) = U_ite(i, 1) + res;
    
            res = h_temp * (F_V(1, i) + h_sq_1*(V_ite(1 ,i+1)+V_ite(1, i-1)+V_ite(2, i)-3*V_ite(1, i))...
                -N*(P_ite(1, i)-P_ite(1, i-1)));
            V_ite(1, i) = V_ite(1, i) + res; 
        end
    
        for i = 3:2:N-1
            res = h_temp * (F_U(i, N) + h_sq_1*(U_ite(i, N-1)+U_ite(i-1, N)+U_ite(i+1, N)-3*U_ite(i, N))...
                    -N*(P_ite(i, N)-P_ite(i-1, N)));
            U_ite(i, N) = U_ite(i, N) + res;
    
            res = h_temp * (F_V(N, i) + h_sq_1*(V_ite(N ,i+1)+V_ite(N, i-1)+V_ite(N-1, i)-3*V_ite(N, i))...
                -N*(P_ite(N, i)-P_ite(N, i-1)));
            V_ite(N, i) = V_ite(N, i) + res; 
        end
    
        % ================== 黑节点更新阶段 ==================
        % 重新计算残差（使用更新后的红节点）
        h_temp = h_sq / 4;
        for i = 2:N
            mod_j = mod(i, 2) + 2;
            for j = mod_j:2:N-1
                res = h_temp * (F_U(i, j) + h_sq_1*(U_ite(i,j+1)+U_ite(i,j-1)+U_ite(i-1,j)+U_ite(i+1,j)-4*U_ite(i,j))...
                    -N*(P_ite(i,j)-P_ite(i-1,j)));
                U_ite(i, j) = U_ite(i, j) + res;
            end
        end
    
        for i = 2:N-1
            mod_j = mod(i, 2) + 2;
            for j = mod_j:2:N
                res = h_temp * (F_V(i, j) + h_sq_1*(V_ite(i,j+1)+V_ite(i,j-1)+V_ite(i-1,j)+V_ite(i+1,j)-4*V_ite(i,j))...
                    -N*(P_ite(i,j)-P_ite(i,j-1)));
                V_ite(i, j) = V_ite(i, j) + res;
            end
        end
        
        h_temp = h_sq / 3;
        %边界处理
        for i = 3:2:N
            res = h_temp * (F_U(i, 1) + h_sq_1*(U_ite(i, 2)+U_ite(i-1, 1)+U_ite(i+1, 1)-3*U_ite(i, 1))...
                    -N*(P_ite(i, 1)-P_ite(i-1, 1)));
            U_ite(i, 1) = U_ite(i, 1) + res;
    
            res = h_temp * (F_V(1, i) + h_sq_1*(V_ite(1 ,i+1)+V_ite(1, i-1)+V_ite(2, i)-3*V_ite(1, i))...
                -N*(P_ite(1, i)-P_ite(1, i-1)));
            V_ite(1, i) = V_ite(1, i) + res; 
        end
    
        for i = 2:2:N
            res = h_temp * (F_U(i, N) + h_sq_1*(U_ite(i, N-1)+U_ite(i-1, N)+U_ite(i+1, N)-3*U_ite(i, N))...
                    -N*(P_ite(i, N)-P_ite(i-1, N)));
            U_ite(i, N) = U_ite(i, N) + res;
    
            res = h_temp * (F_V(N, i) + h_sq_1*(V_ite(N ,i+1)+V_ite(N, i-1)+V_ite(N-1, i)-3*V_ite(N, i))...
                -N*(P_ite(N, i)-P_ite(N, i-1)));
            V_ite(N, i) = V_ite(N, i) + res; 
        end

        %红格子
        %(2:N-1, 2:N-1)红格+(1, 2:2:N-1)+(N, 3:2:N-1)+(2:2:N-1, 1)+(3:2:N-1, N)+(N, 1)+(1, N)
        %更新P--red-inner
        %更新U--red-inner
        %更新V--red-inner
         for i = 2:N-1
            mod_j = 2 + mod(i+1, 2);
            for j = mod_j:2:N-1
                res = 0.25*(-N*(U_ite(i+1, j)-U_ite(i, j)+V_ite(i, j+1)-V_ite(i, j))-D(i, j));
                P_ite(i, j) = P_ite(i, j) + 4 * res;
                P_ite(i-1, j) = P_ite(i-1, j) - res;
                P_ite(i, j-1) = P_ite(i, j-1) - res;
                P_ite(i+1, j) = P_ite(i+1, j) - res;
                P_ite(i, j+1) = P_ite(i, j+1) - res;
                res = h * res;
                U_ite(i, j) = U_ite(i, j) - res;
                U_ite(i+1, j) = U_ite(i+1, j) + res;
                V_ite(i, j) = V_ite(i, j) - res;
                V_ite(i, j+1) = V_ite(i, j+1) + res;
            end
        end   
    
        %更新边界
        %(1, 2:2:N-1)->U = 0
        for i = 2:2:N-1
            res = 1/3*(-N*(U_ite(2, i)+V_ite(1, i+1)-V_ite(1, i))-D(1, i));
            P_ite(1, i) = P_ite(1, i) + 3 * res;
            P_ite(1, i-1) = P_ite(1, i-1) - res;
            P_ite(1, i+1) = P_ite(1, i+1) - res;
            P_ite(2, i) = P_ite(2, i) - res;
            res = res * h;
            U_ite(2, i) = U_ite(2, i) + res;
            V_ite(1, i) = V_ite(1, i) - res;
            V_ite(1, i+1) = V_ite(1, i+1) +res;
        end
    
        %(N, 3:2:N-1)->U = 0 
        for i = 3:2:N-1
            res = 1/3*(-N*(-U_ite(N, i)+V_ite(N, i+1)-V_ite(N, i))-D(N, i));
            P_ite(N, i) = P_ite(N, i) + 3 * res;
            P_ite(N, i-1) = P_ite(N, i-1) - res;
            P_ite(N, i+1) = P_ite(N, i+1) - res;
            P_ite(N-1, i) = P_ite(N-1, i) - res;
            res = res * h;
            U_ite(N, i) = U_ite(N, i) - res;
            V_ite(N, i) = V_ite(N, i) - res;
            V_ite(N, i+1) = V_ite(N, i+1) +res;
        end
    
        %(2:2:N-1, 1)->V = 0
        for i = 2:2:N-1
            res = 1/3*(-N*(U_ite(i+1, 1)-U_ite(i, 1)+V_ite(i, 2))-D(i, 1));
            P_ite(i, 1) = P_ite(i, 1) + 3*res;
            P_ite(i-1, 1) = P_ite(i-1, 1) - res;
            P_ite(i+1, 1) = P_ite(i+1, 1) - res;
            P_ite(i, 2) = P_ite(i, 2) - res;
            res = res * h;
            U_ite(i, 1) = U_ite(i, 1) - res;
            U_ite(i+1, 1) = U_ite(i+1, 1) + res;
            V_ite(i, 2) = V_ite(i, 2) + res;
        end
    
        %(3:2:N-1, N)->V = 0
        for i = 3:2:N-1
            res = 1/3*(-N*(U_ite(i+1, N)-U_ite(i, N)-V_ite(i, N))-D(i, N));
            P_ite(i, N) = P_ite(i, N) + 3*res;
            P_ite(i-1, N) = P_ite(i-1, N) - res;
            P_ite(i+1, N) = P_ite(i+1, N) - res;
            P_ite(i, N-1) = P_ite(i, N-1) - res;
            res = res * h;
            U_ite(i, N) = U_ite(i, N) - res;
            U_ite(i+1, N) = U_ite(i+1, N) + res;
            V_ite(i, N) = V_ite(i, N) - res;
        end
        
        %(N, 1)
        res = 0.5*(-N*(-U_ite(N, 1)+V_ite(N, 2))-D(N, 1));
        P_ite(N, 1) = P_ite(N, 1) + 2*res;
        P_ite(N, 2) = P_ite(N, 2) - res;
        P_ite(N-1, 1) = P_ite(N-1, 1) - res;
        res = h*res;
        U_ite(N, 1) = U_ite(N, 1) - res;
        V_ite(N, 2) = V_ite(N, 2) + res;    
    
    
        %(1, N)  
        res = 0.5*(-N*(U_ite(2, N)-U_ite(1, N)+V_ite(1, N+1)-V_ite(1, N))-D(1, N));
        P_ite(1, N) = P_ite(1, N) + 2*res;
        P_ite(2, N) = P_ite(2, N) - res;
        P_ite(1, N-1) = P_ite(1, N-1) - res;
        res = h*res;
        U_ite(2, N) = U_ite(2, N) + res;
        V_ite(1, N) = V_ite(1, N) - res;    
    
        %黑格子
        %(2:N-1, 2:N-1)黑格+(1, 3:2:N-1)+(N, 2:2:N-2)+(3:2:N-1, 1)+(2:2:N-2, N)
        %+(1, 1)+(N, N)
        %更新P--balck-inner
        %更新U--black-inner
        %更新V--black-inner
         for i = 2:N-1
            mod_j = 2 + mod(i, 2);
            for j = mod_j:2:N-1
                res = 0.25*(-N*(U_ite(i+1, j)-U_ite(i, j)+V_ite(i, j+1)-V_ite(i, j))-D(i, j));
                P_ite(i, j) = P_ite(i, j) + 4 * res;
                P_ite(i-1, j) = P_ite(i-1, j) - res;
                P_ite(i, j-1) = P_ite(i, j-1) - res;
                P_ite(i+1, j) = P_ite(i+1, j) - res;
                P_ite(i, j+1) = P_ite(i, j+1) - res;
                res = h * res;
                U_ite(i, j) = U_ite(i, j) - res;
                U_ite(i+1, j) = U_ite(i+1, j) + res;
                V_ite(i, j) = V_ite(i, j) - res;
                V_ite(i, j+1) = V_ite(i, j+1) + res;
            end
        end 
    
        %(1, 3:2:N-1)->U = 0
        for i = 3:2:N-1
            res = 1/3*(-N*(U_ite(2, i)+V_ite(1, i+1)-V_ite(1, i))-D(1, i));
            P_ite(1, i) = P_ite(1, i) + 3 * res;
            P_ite(1, i-1) = P_ite(1, i-1) - res;
            P_ite(1, i+1) = P_ite(1, i+1) - res;
            P_ite(2, i) = P_ite(2, i) - res;
            res = res * h;
            U_ite(2, i) = U_ite(2, i) + res;
            V_ite(1, i) = V_ite(1, i) - res;
            V_ite(1, i+1) = V_ite(1, i+1) +res;
        end
    
        %(N, 2:2:N-2)->U = 0    
        for i = 2:2:N-1
            res = 1/3*(-N*(-U_ite(N, i)+V_ite(N, i+1)-V_ite(N, i))-D(N, i));
            P_ite(N, i) = P_ite(N, i) + 3 * res;
            P_ite(N, i-1) = P_ite(N, i-1) - res;
            P_ite(N, i+1) = P_ite(N, i+1) - res;
            P_ite(N-1, i) = P_ite(N-1, i) - res;
            res = res * h;
            U_ite(N, i) = U_ite(N, i) - res;
            V_ite(N, i) = V_ite(N, i) - res;
            V_ite(N, i+1) = V_ite(N, i+1) +res;
        end
    
        %(3:2:N-1, 1)->V = 0    
        for i = 3:2:N-1
            res = 1/3*(-N*(U_ite(i+1, 1)-U_ite(i, 1)+V_ite(i, 2))-D(i, 1));
            P_ite(i, 1) = P_ite(i, 1) + 3*res;
            P_ite(i-1, 1) = P_ite(i-1, 1) - res;
            P_ite(i+1, 1) = P_ite(i+1, 1) - res;
            P_ite(i, 2) = P_ite(i, 2) - res;
            res = res * h;
            U_ite(i, 1) = U_ite(i, 1) - res;
            U_ite(i+1, 1) = U_ite(i+1, 1) + res;
            V_ite(i, 2) = V_ite(i, 2) + res;
        end
    
    %     %(2:2:N-2, N)->V = 0    
        for i = 2:2:N-1
            res = 1/3*(-N*(U_ite(i+1, N)-U_ite(i, N)-V_ite(i, N))-D(i, N));
            P_ite(i, N) = P_ite(i, N) + 3*res;
            P_ite(i-1, N) = P_ite(i-1, N) - res;
            P_ite(i+1, N) = P_ite(i+1, N) - res;
            P_ite(i, N-1) = P_ite(i, N-1) - res;
            res = res * h;
            U_ite(i, N) = U_ite(i, N) - res;
            U_ite(i+1, N) = U_ite(i+1, N) + res;
            V_ite(i, N) = V_ite(i, N) - res;
        end
    
        %(1, 1)  
        res = 0.5*(-N*(U_ite(2, 1)+V_ite(1, 2))-D(1, 1));
        P_ite(1, 1) = P_ite(1, 1) + 2*res;
        P_ite(1, 2) = P_ite(1, 2) - res;
        P_ite(2, 1) = P_ite(2, 1) - res;
        res = h*res;
        U_ite(2, 1) = U_ite(2, 1) + res;
        V_ite(1, 2) = V_ite(1, 2) + res; 
    
        %(N, N)
        res = 0.5*(-N*(U_ite(N+1, N)-U_ite(N, N)+V_ite(N, N+1)-V_ite(N, N))-D(N, N));
        P_ite(N, N) = P_ite(N, N) + 2*res;
        P_ite(N, N-1) = P_ite(N, N-1) - res;
        P_ite(N-1, N) = P_ite(N-1, N) - res;
        res = h*res;
        U_ite(N, N) = U_ite(N, N) - res;
        V_ite(N, N) = V_ite(N, N) - res; 
    end
end

% 辅助函数：创建速度场U的红黑掩码（(N+1)xN网格）
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

function [red_mask, black_mask] = create_red_mask_lattice(N)
    [I, J] = ndgrid(1:N, 1:N);
    I = gpuArray(I);
    J = gpuArray(J);
    red_mask = mod(I + J, 2) == 1;
    red_mask([1, N], :) = false;        
    red_mask(:, [1, N]) = false;

    black_mask = ~red_mask;
    black_mask([1, N], :) = false;     
    black_mask(:, [1, N]) = false;
end
