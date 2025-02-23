% 数据
N = [64, 128, 256, 512, 1024, 2048, 4096];
error = [0.0014951, 0.00037363, 0.000093399, 0.000023349, 0.0000058372, 0.0000014593, 0.00000036483];

for i = 1:6
    disp(error(1, i) / error(1, i+1))
end

% 绘制对数-对数图
figure;
loglog(N, error, '-o', 'LineWidth', 1);
grid on;

% 设置坐标轴标签
xlabel('N');
ylabel('误差');

% 设置标题
title('误差与N的关系');

% 设置图例
legend('误差');