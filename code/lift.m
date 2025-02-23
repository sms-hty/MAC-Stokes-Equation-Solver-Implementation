%提升算子
function [U_lift, V_lift, P_lift] = lift(U, V, P, N, device)
    if device == 'gpu'
        U_lift = gpuArray.zeros(N*2 + 1, N*2);
        V_lift = gpuArray.zeros(N*2, N*2 + 1);
        P_lift = gpuArray.zeros(N*2, N*2);
%         copy_matrix_1 = gpuArray([1 1; 1 1]);
%         copy_matrix_col = gpuArray([1; 1]);
%         copy_matrix_row = gpuArray([1 1]);
    else
        U_lift = zeros(N*2 + 1, N*2);
        V_lift = zeros(N*2, N*2 + 1);
        P_lift = zeros(N*2, N*2);
%         copy_matrix_1 = [1 1; 1 1];
%         copy_matrix_col = [1; 1];
%         copy_matrix_row = [1 1];
    end
%    P_lift = kron(P, copy_matrix_1);
    P_lift(1:2:end, 1:2:end) = P;
    P_lift(1:2:end, 2:2:end) = P;
    P_lift(2:2:end, 1:2:end) = P;
    P_lift(2:2:end, 2:2:end) = P;

%     U_lift(1:2:(N*2+1), 1:(N*2)) = kron(U, copy_matrix_row);
    U_lift(1:2:end, 1:2:(N*2)) = U;
    U_lift(1:2:end, 2:2:(N*2)) = U;

%    U_lift(2:2:N*2, 1:N*2) = kron((U(1:N, 1:N) + U(2:N+1, 1:N)), copy_matrix_row / 2);
    U_lift(2:2:N*2, 1:2:N*2) = (U(1:N, 1:N) + U(2:N+1, 1:N)) / 2;
    U_lift(2:2:N*2, 2:2:N*2) = U_lift(2:2:N*2, 1:2:N*2);


%     V_lift(1:N*2, 1:2:N*2+1) = kron(V, copy_matrix_col);
    V_lift(1:2:N*2, 1:2:end) = V;
    V_lift(2:2:N*2, 1:2:end) = V;
%     
%     V_lift(1:N*2, 2:2:N*2) = kron((V(1:N, 1:N) + V(1:N, 2:N+1)), copy_matrix_col / 2);
    V_lift(2:2:N*2, 2:2:N*2) = (V(:, 1:N) + V(:, 2:N+1)) / 2;
    V_lift(1:2:N*2, 2:2:N*2) = V_lift(2:2:N*2, 2:2:N*2);
end