clear
Table = readmatrix("generated_fake_data_correlated.csv");
MRI_flag = 1;
%%
errorLog = {};  % 创建空cell数组保存错误信息
logIdx = 1;      % 初始化日志索引
output = cell(1,100);
labels = zeros(1,100);
for PATIENT_NO = 1:100
    try
        FakeParamsT = Table(PATIENT_NO,1:34);
        [label,o_vals] = simulate_from_python(FakeParamsT);
        output{PATIENT_NO} = o_vals;
        labels(PATIENT_NO) = label;
    catch ME
        disp({ME.message num2str(PATIENT_NO)})
        errorLog{logIdx,1} = PATIENT_NO;
        errorLog{logIdx,2} = ME.message;
        logIdx = logIdx + 1;
    end
end

length(labels(labels == 1))
%%
T1 = readtable("Training.xlsx");
% 1. 过滤掉空 struct
validIdx = cellfun(@(s) isstruct(s) && ~isempty(fieldnames(s)), output);
validOutput = output(validIdx);
labels_clean = labels(validIdx).';

% 2. 获取字段名
fieldNames = fieldnames(validOutput{1});
numPatients = numel(validOutput);
T = table();

% 3. 提取字段（逐列）
for f = 1:numel(fieldNames)
    fname = fieldNames{f};
    % 先试图用标量提取
    vals = cellfun(@(s) s.(fname), validOutput, 'UniformOutput', false);
    
    % 检查是不是标量（如果是标量，就转换成列向量）
    if all(cellfun(@isscalar, vals)) && all(cellfun(@isnumeric, vals))
        T.(fname) = cell2mat(vals(:));  % 保证是列向量
    else
        % 否则用 cell 存，确保列方向
        T.(fname) = vals(:);  % 转成列cell
    end
end

%%
fieldsT1 = T1.Properties.VariableNames;
numFields = numel(fieldsT1);
plotsPerFigure = 12;

for figIdx = 1:ceil(numFields / plotsPerFigure)
    figure('Name', sprintf('Histogram Group %d', figIdx), 'Color', 'w');
    
    for plotIdx = 1:plotsPerFigure
        fieldIdx = (figIdx - 1) * plotsPerFigure + plotIdx;
        if fieldIdx > numFields
            break;
        end

        field_S = fieldsT1{fieldIdx};              % e.g. 'SBP_S'
        field = regexprep(field_S, '_S$', '');


        % 跳过不存在字段
        if ~ismember(field, T.Properties.VariableNames)
            continue;
        end

        % 取数据
        data_all = T.(field);
        data_sub = T1.(field_S);

        % 跳过非数值字段
        if ~isnumeric(data_all) || ~isnumeric(data_sub)
            continue;
        end

        % 设置 bin 边界（以 T1 为基础）
        minVal = min(data_sub);
        maxVal = max(data_sub);
        if minVal == maxVal
            minVal = minVal - 0.5;
            maxVal = maxVal + 0.5;
        end
        binEdges = linspace(minVal, maxVal, 15);

        % 绘图
        subplot(3, 4, plotIdx);
        hold on;
        histogram(data_all, binEdges, 'FaceColor', [0.2 0.5 0.8], ...
                  'EdgeColor', 'k', 'FaceAlpha', 0.7, 'Normalization', 'probability');
        histogram(data_sub, binEdges, 'FaceColor', [0.9 0.3 0.3], ...
                  'EdgeColor', 'k', 'FaceAlpha', 0.5, 'Normalization', 'probability');
        title(field, 'Interpreter', 'none');
        xlabel(field);
        ylabel('Frequency');
        legend({'VAE', 'Real'}, 'Location', 'northeast');
        box on;
    end
end
