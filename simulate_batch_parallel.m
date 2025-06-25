function labels = simulate_batch_parallel(inputData)
% inputData: N x M, 每行一组参数
% labels: N x 1，输出标签
N = size(inputData, 1);
labels = zeros(N, 1);
% 并行处理每一行
if isempty(gcp('nocreate'))
    parpool(12);
    parfor i = 1:N
        labels(i) = simulate_from_python(inputData(i, :));
    end
end
% Delete the parallel pool
delete(gcp("nocreate"));
end
