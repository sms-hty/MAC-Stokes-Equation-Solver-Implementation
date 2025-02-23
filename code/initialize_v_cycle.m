function [F_U, F_V, U0, V0] = initialize_v_cycle(N)
h = 1 / N;
F_U = zeros(N+1,N);
F_V = zeros(N,N+1);
U0 = zeros(N+1,N);
V0 = zeros(N,N+1);;
for i=1:N+1
    for j=1:N
        x = (i-1)*h;
        y = 1/2*h + (j-1)*h;
        U0(i, j) = (1 - cos(2 * pi * x)) * sin(2 * pi * y);
    end
end
for j=1:N+1
    for i=1:N
        x = 1/2 * h + (i - 1) * h;
        y = (j - 1) * h;
        V0(i, j) = -(1 - cos(2 * pi * y)) * sin(2 * pi * x);
    end
end
for j=2:N-1
    for i=2:N
        x = (i - 1) * h;
        y =(j - 1/2) * h;
        F_U(i, j) = -4 * pi^2 * (2 * cos(2 * pi * x) -1) * sin(2 * pi * y) + x^2;
    end
end
for i=2:N
    y = 1/2 * h;
    x = (i - 1) * h;
    F_U(i, 1) =-4 * pi^2 * (2 * cos(2 * pi * x) -1) * sin(2 * pi * y) + x^2 + 1/h * (2 * pi *(cos(2 * pi * x) - 1));
    y = (N - 1/2) * h;
    x = (i - 1) * h;
    F_U(i, N)=-4 * pi^2 * (2 * cos(2 * pi * x) - 1) * sin(2 * pi * y) + x^2 + 1/h * (-2 * pi * (cos(2 * pi * x) - 1));
end

for i=2:N-1
    for j=2:N
        x = (i - 1/2) * h;
        y = (j - 1) * h;
        F_V(i, j) = 4 * pi^2 * (2 * cos(2 * pi * y) - 1) * sin(2 * pi * x);
    end
end
for j=2:N
    i = 1;
    x = (i - 1/2) * h;
    y = (j - 1) * h;
    F_V(i, j) = 4 * pi^2 * (2 * cos(2 * pi * y) - 1) * sin(2 * pi * x) + 1/h * (-2 * pi * (cos(2 * pi * y) - 1));
    i = N;
    x = (i - 1/2) * h;
    y = (j - 1) * h;
    F_V(i, j) = 4 * pi^2 * (2 * cos(2 * pi * y) -1) * sin(2 * pi * x) + 1/h * (2 * pi * (cos(2 * pi * y) - 1));
end
end