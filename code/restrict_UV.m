function [U_res, V_res] = restrict_UV(U, V, N, device)
    if device == 'gpu'
        U_res = gpuArray.zeros(N/2 + 1, N/2);
        V_res = gpuArray.zeros(N/2, N/2 + 1);
        %P_res = gpuArray.zeros(N/2, N/2);
    else
        U_res = zeros(N/2 + 1, N/2);
        V_res = zeros(N/2, N/2 + 1);
        %P_res = zeros(N/2, N/2);
    end
    %P_res
%     P_res = (P(1:2:N, 1:2:N) + ...
%              P(1:2:N, 2:2:N) + ...
%              P(2:2:N, 1:2:N) + ...
%              P(2:2:N, 2:2:N)) / 4;
    %U_res
    U_res(2:N/2, 1:N/2) = (U(3:2:N, 1:2:N) + U(3:2:N, 2:2:N)) / 4 + ...
                          (U(2:2:N-2, 1:2:N) + U(2:2:N-2, 2:2:N) +...
                           U(4:2:N, 1:2:N) + U(4:2:N, 2:2:N)) / 8 ;
    U_res(1, 1:N/2)  = (U(1, 1:2:N) + U(1, 2:2:N)) / 2;
    U_res(N/2 + 1, 1:N/2) = (U(N + 1, 1:2:N) + U(N + 1, 2:2:N)) / 2;
    %V_res
    V_res(1:N/2, 2:N/2) = (V(1:2:N, 3:2:N) + V(2:2:N, 3:2:N)) / 4 +...
                          (V(1:2:N, 2:2:N-2) + V(1:2:N, 4:2:N) + ...
                           V(2:2:N, 2:2:N-2) + V(2:2:N, 4:2:N)) / 8;
    V_res(1:N/2, 1) = (V(1:2:N, 1) + V(2:2:N, 1)) / 2;
    V_res(1:N/2, N/2+1) = (V(1:2:N, N+1) + V(2:2:N, N+1)) / 2;
end