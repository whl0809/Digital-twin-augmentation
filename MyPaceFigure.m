%% NT-proBNP 相关性：9 个classifier × (治疗/对照)；每组一个 3×3 图阵
clear; clc;

subset = 2;% 1 for 6k 2 for 25K
% 读取数据
T1 = readtable('CorrelationT.xlsx','VariableNamingRule','preserve','Sheet',subset+1);
CutoffT = readtable("optimal_thresholds_summary.csv");
if subset == 1
    Traincutoff = CutoffT.subset_threshold;
else
    Traincutoff = CutoffT.full_threshold;
end

varNames = T1.Properties.VariableNames;

% 列索引
clf_cols = 1:9;           % 9个classifier概率
grp_col  = 10;            % 分组：治疗=1，对照=0
BNP0_col = 11;            % baseline NT-proBNP
BNP1m_col= 12;            % 1个月 NT-proBNP

% 提取数据
scores   = T1{:, clf_cols};     % N × 9
group    = T1{:, grp_col};      % N × 1
BNP0     = T1{:, BNP0_col};     % N × 1
BNP1m    = T1{:, BNP1m_col};    % N × 1

% 基本校验
validBNP = BNP0 > 0 & BNP1m > 0 & isfinite(BNP0) & isfinite(BNP1m);
if any(~validBNP)
    warning('发现 %d 个 BNP<=0 或非有限值样本，将在分析中忽略这些样本。', sum(~validBNP));
end

% ΔBNP（下降为负，表示改善）
dBNP = BNP1m - BNP0;

% —— BNP fractional change（与 myPACE 口径一致）——
% 注意：这里改善=下降 → 数值更"负"
dFracBNP = (BNP1m - BNP0) ./ BNP0;   % 单位：比值（fraction）
yLabelBNP_frac = 'Fractional change in NT-proBNP ((1m - baseline) / baseline)';


% "始终正常"标记：基线与1月都 <125 pg/mL
isAlwaysNormal = (BNP0 < 125) & (BNP1m < 125);

% 颜色
c_treat  = [0.10 0.45 0.85];  % 蓝：治疗组
c_ctrl   = [0.85 0.25 0.10];  % 红：对照组

% 结果表
res = table(); rowCounter = 0;

%% --------- 绘图与统计：治疗组（3×3）---------
fig1 = figure('Color','w','Name','Treatment (1)');
tlo1 = tiledlayout(fig1,3,3,'TileSpacing','compact','Padding','compact');
title(tlo1,'Treatment group (label=1): Classifier vs \Delta NT-proBNP');

for i = 1:numel(clf_cols)
    clf_name = varNames{clf_cols(i)};
    x = scores(:, i);

    % 有效样本（治疗组）
    valid = validBNP & group==1 & ~isnan(x) & ~isnan(dBNP);

    idx_all   = find(valid);
    idx_info  = idx_all(~isAlwaysNormal(idx_all));  % 非"始终<125"
    idx_noise = idx_all( isAlwaysNormal(idx_all));  % "始终<125"

    nexttile; hold on; box on; grid on;

    % 有意义（圆点）
    if ~isempty(idx_info)
        scatter(x(idx_info), dBNP(idx_info), 36, 'o', ...
            'MarkerEdgeColor', c_treat, 'MarkerFaceColor', c_treat, ...
            'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
    end
    % 始终<125（叉号）
    if ~isempty(idx_noise)
        scatter(x(idx_noise), dBNP(idx_noise), 36, 'x', ...
            'MarkerEdgeColor', c_treat, 'LineWidth', 1.4);
    end

    xlabel(clf_name, 'Interpreter','none');
    if i==1 || i==4 || i==7
        ylabel('\Delta NT-proBNP (baseline - 1m)');
    end
    title(clf_name, 'Interpreter','none','FontWeight','normal');

    % Spearman：全部 vs 排除<125
    x_all = x(valid);        y_all = dBNP(valid);
    [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

    keep = valid & ~isAlwaysNormal;
    x_fil = x(keep);         y_fil = dBNP(keep);
    if numel(x_fil) >= 3
        [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
    else
        rho_fil = NaN; p_fil = NaN;
    end


    % 显著性阈值（可改）
    alphaLevel = 0.05;

    % 小注释内容
    txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<125}=%.3f, p=%.3g', ...
        rho_all, p_all, rho_fil, p_fil);

    % 是否显著：任意一个p满足阈值
    isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);

    % 颜色：显著=红色，否则=灰色
    if isSig
        txtColor = [0.85 0.15 0.15];  % 红
    else
        txtColor = [0.2 0.2 0.2];     % 灰
    end


    % 小注释（右下角）

    xlim_curr = xlim; ylim_curr = ylim;
    text(xlim_curr(1)+0.22*range(xlim_curr), ylim_curr(2)-0.65*range(ylim_curr), txt, ...
        'Color',txtColor, 'FontSize',9, 'Interpreter','tex');

    % 结果表写入
    rowCounter = rowCounter + 1;
    res(rowCounter,:) = table( ...
        string(clf_name), "Treatment", ...
        sum(valid), rho_all, p_all, ...
        sum(keep),  rho_fil, p_fil, ...
        'VariableNames', {'Classifier','Group','N_all','Rho_all','P_all','N_excl125','Rho_excl125','P_excl125'} ...
        );
end

% 图例（统一放在整体标题下方）
lg1 = legend({'Informative (not both <125)','Always <125 (×)'}, 'Orientation','horizontal');
lg1.Layout.Tile = 'south';

%% --------- 绘图与统计：对照组（3×3）---------
fig0 = figure('Color','w','Name','Control (0)');
tlo0 = tiledlayout(fig0,3,3,'TileSpacing','compact','Padding','compact');
title(tlo0,'Control group (label=0): Classifier vs \Delta NT-proBNP');

for i = 1:numel(clf_cols)
    clf_name = varNames{clf_cols(i)};
    x = scores(:, i);

    % 有效样本（对照组）
    valid = validBNP & group==0 & ~isnan(x) & ~isnan(dBNP);

    idx_all   = find(valid);
    idx_info  = idx_all(~isAlwaysNormal(idx_all));
    idx_noise = idx_all( isAlwaysNormal(idx_all));

    nexttile; hold on; box on; grid on;

    % 有意义（圆点）
    if ~isempty(idx_info)
        scatter(x(idx_info), dBNP(idx_info), 36, 'o', ...
            'MarkerEdgeColor', c_ctrl, 'MarkerFaceColor', c_ctrl, ...
            'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
    end
    % 始终<125（叉号）
    if ~isempty(idx_noise)
        scatter(x(idx_noise), dBNP(idx_noise), 36, 'x', ...
            'MarkerEdgeColor', c_ctrl, 'LineWidth', 1.4);
    end

    xlabel(clf_name, 'Interpreter','none');
    if i==1 || i==4 || i==7
        ylabel('\Delta NT-proBNP (baseline - 1m)');
    end
    title(clf_name, 'Interpreter','none','FontWeight','normal');

    % Spearman：全部 vs 排除<125
    x_all = x(valid);        y_all = dBNP(valid);
    [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

    keep = valid & ~isAlwaysNormal;
    x_fil = x(keep);         y_fil = dBNP(keep);
    if numel(x_fil) >= 3
        [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
    else
        rho_fil = NaN; p_fil = NaN;
    end


    % 显著性阈值（可改）
    alphaLevel = 0.05;

    % 小注释内容
    txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<125}=%.3f, p=%.3g', ...
        rho_all, p_all, rho_fil, p_fil);

    % 是否显著：任意一个p满足阈值
    isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);

    % 颜色：显著=红色，否则=灰色
    if isSig
        txtColor = [0.85 0.15 0.15];  % 红
    else
        txtColor = [0.2 0.2 0.2];     % 灰
    end


    % 放置位置（你给的偏移）
    xlim_curr = xlim; ylim_curr = ylim;
    text(xlim_curr(1)+0.22*range(xlim_curr), ...
        ylim_curr(2)-0.65*range(ylim_curr), ...
        txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');


    % 结果表写入
    rowCounter = rowCounter + 1;
    res(rowCounter,:) = table( ...
        string(clf_name), "Control", ...
        sum(valid), rho_all, p_all, ...
        sum(keep),  rho_fil, p_fil, ...
        'VariableNames', {'Classifier','Group','N_all','Rho_all','P_all','N_excl125','Rho_excl125','P_excl125'} ...
        );
end

lg0 = legend({'Informative (not both <125)','Always <125 (×)'}, 'Orientation','horizontal');
lg0.Layout.Tile = 'south';

%% 汇总结果
disp('===== Spearman 结果汇总（每个 classifier × 组；含全部 与 排除始终<125）=====');
disp(res);

%% ========= 基于训练cutoff，把9个分数二分为 High(1)/Low(0) =========
probNames = varNames(clf_cols);                  % 概率列的真实列名
cutvec = resolve_cutvec_positional(Traincutoff, numel(clf_cols));  % 直接按顺序取

% 二分，高=1 低=0
HL = bsxfun(@ge, scores, cutvec);               % N×9 logical

%% ========= 在治疗组与对照组分别比较 High vs Low 的 ΔNT-proBNP =========
% 提醒：你现在定义的是 dBNP = BNP1m - BNP0（负值=改善）
yLabelBNP = '\Delta NT-proBNP (1m - baseline)';

alphaLevel = 0.05;

resHL_T = compare_high_low_3x3(HL, group, 1, dFracBNP, validBNP, isAlwaysNormal, ...
    clf_cols, varNames, yLabelBNP_frac, 'Treatment (label=1): High vs Low (fixed cutoff; fractional change)', ...
    [0.10 0.45 0.85], alphaLevel);

resHL_C = compare_high_low_3x3(HL, group, 0, dFracBNP, validBNP, isAlwaysNormal, ...
    clf_cols, varNames, yLabelBNP_frac, 'Control (label=0): High vs Low (fixed cutoff; fractional change)', ...
    [0.85 0.25 0.10], alphaLevel);


% 合并与保存结果
resHL_all = [resHL_T; resHL_C];
disp('===== High vs Low (fixed training cutoff) on ΔNT-proBNP =====');
disp(resHL_all);

%% =================== 辅助函数：从表中解析cutoff ===================
function cutvec = resolve_cutvec_positional(Traincutoff, M)
% 只按顺序取前 M 个数，不看列名。
% 兼容 1×M（横向）或 M×1（纵向），也兼容含多列但前 M 个是阈值的情况。

% 只保留数值型列，避免字符串/名字干扰
try
    numcols = varfun(@isnumeric, Traincutoff, 'OutputFormat','uniform');
    data = Traincutoff{:, numcols};
catch
    % 老版本 MATLAB 没有上面的写法就退化成直接取全部
    data = Traincutoff(:, :);
end

if ~isnumeric(data)
    error('Traincutoff 表含非数值列；请确保阈值是数值，并把非数值列去掉或放在后面。');
end

vals = data(:)';  % 按列展开成行向量；1×M 或 M×1 都 OK
if numel(vals) < M
    error('阈值数量不足：需要 %d 个，实际只有 %d 个。', M, numel(vals));
end
cutvec = vals(1:M);  % 取前 M 个，按顺序对应 9 个分类器
end

%% ============== 辅助函数：3×3箱线图 + ranksum 检验（High vs Low） ==============
function resTbl = compare_high_low_3x3(HL, group, whichGroup, ...
    dY, validPair, isAlwaysTrivial, clf_cols, varNames, yLabel, figTitle, colorUse, alphaLevel)

fig = figure('Color','w','Name',figTitle);
tlo = tiledlayout(fig,3,3,'TileSpacing','compact','Padding','compact');
title(tlo, figTitle, 'Interpreter','none');

resTbl = table();
rc = 0;

for k = 1:numel(clf_cols)
    clf_name = varNames{clf_cols(k)};

    % 有效样本
    valid = (group==whichGroup) & validPair & ~isnan(dY);

    % High / Low（基于固定 cutoff -> HL）
    high   = HL(:, k) & valid;     % score >= cutoff
    low    = ~HL(:, k) & valid;    % score <  cutoff

    % 排除"trivial"后的版本（比如 <125 的过滤）
    keep_ex = valid & ~isAlwaysTrivial;
    high_ex = HL(:, k) & keep_ex;
    low_ex  = ~HL(:, k) & keep_ex;

    % ---- 作图：数值分组 1=Low, 2=High ----
    nexttile; hold on; box on; grid on;

    grp  = [ones(sum(low),1); 2*ones(sum(high),1)];
    vals = [dY(low); dY(high)];

    if ~isempty(vals)
        try
            boxchart(grp, vals, 'BoxFaceColor', colorUse, 'MarkerStyle','none');
        catch
            boxplot(vals, grp, 'Positions',[1 2], 'Colors', colorUse, 'Labels', {'Low','High'});
        end
    else
        set(gca,'XTick',[1 2],'XTickLabel',{'Low','High'});
    end

    % 抖动散点
    if sum(low)>0
        scatter(1 + 0.06*randn(sum(low),1), dY(low), 16, ...
            'MarkerEdgeColor', colorUse, 'MarkerFaceColor','w', ...
            'MarkerFaceAlpha',0.7,'MarkerEdgeAlpha',0.7);
    end
    if sum(high)>0
        scatter(2 + 0.06*randn(sum(high),1), dY(high), 16, ...
            'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
            'MarkerFaceAlpha',0.25,'MarkerEdgeAlpha',0.85);
    end

    xlim([0.5 2.5]);
    set(gca,'XTick',[1 2],'XTickLabel',{'Low','High'});
    xlabel(sprintf('%s  (Low vs High)', clf_name), 'Interpreter','none');
    if k==1 || k==4 || k==7
        ylabel(yLabel, 'Interpreter','none');
    end
    title(clf_name, 'Interpreter','none','FontWeight','normal');

    % ---- 统计：ranksum（保留）+ 强制 Welch t 检验 ----
    p_all = NaN; p_ex = NaN;
    med_low_all  = NaN; med_high_all  = NaN;
    med_low_ex   = NaN; med_high_ex   = NaN;

    if sum(low)>=1 && sum(high)>=1
        p_all         = ranksum(dY(low),  dY(high),  'method','approx');  % pk
        med_low_all   = median(dY(low),   'omitnan');
        med_high_all  = median(dY(high),  'omitnan');
    end
    if sum(low_ex)>=1 && sum(high_ex)>=1
        p_ex          = ranksum(dY(low_ex), dY(high_ex), 'method','approx');
        med_low_ex    = median(dY(low_ex),  'omitnan');
        med_high_ex   = median(dY(high_ex), 'omitnan');
    end

    % —— 强制做 t 检验（Welch, Vartype='unequal'），不再检查正态 ——
    p_t_all = NaN;  mdiff_all = NaN;   % pt（High - Low 的均值差）
    p_t_ex  = NaN;  mdiff_ex  = NaN;

    if sum(low)>=2 && sum(high)>=2
        [~, p_t_all] = ttest2(dY(high), dY(low), 'Vartype','unequal');  % pt
        mdiff_all = mean(dY(high),'omitnan') - mean(dY(low),'omitnan'); % High-Low
    end
    if sum(low_ex)>=2 && sum(high_ex)>=2
        [~, p_t_ex] = ttest2(dY(high_ex), dY(low_ex), 'Vartype','unequal');
        mdiff_ex = mean(dY(high_ex),'omitnan') - mean(dY(low_ex),'omitnan');
    end

    % ---- 注释：显示 N、pk（ranksum）、pt（t检验） ----
    isSig = ( ~isnan(p_all)   && p_all   < alphaLevel ) || ...
        ( ~isnan(p_ex)    && p_ex    < alphaLevel ) || ...
        ( ~isnan(p_t_all) && p_t_all < alphaLevel ) || ...
        ( ~isnan(p_t_ex)  && p_t_ex  < alphaLevel );
    txtColor = isSig*[0.85 0.15 0.15] + (~isSig)*[0.2 0.2 0.2];

    txt1 = sprintf('N_all L=%d,H=%d | pk=%.3g  pt=%.3g', sum(low), sum(high), p_all, p_t_all);
    txt2 = sprintf('N_ex  L=%d,H=%d | pk=%.3g  pt=%.3g', sum(low_ex), sum(high_ex), p_ex,  p_t_ex);
    txt3 = sprintf('Median Δ (all): L=%.2f  H=%.2f', med_low_all,  med_high_all);
    txt4 = sprintf('Mean   Δ (all): H-L=%.2f',        mdiff_all);

    xlim_curr = xlim; ylim_curr = ylim;
    y0 = ylim_curr(2)-0.08*range(ylim_curr);
    text(xlim_curr(1)+0.05*range(xlim_curr), y0,                      txt1, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');
    text(xlim_curr(1)+0.05*range(xlim_curr), y0-0.06*range(ylim_curr), txt2, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');
    text(xlim_curr(1)+0.05*range(xlim_curr), y0-0.12*range(ylim_curr), txt3, 'Color',[0.25 0.25 0.25], 'FontSize',8, 'Interpreter','none');
    text(xlim_curr(1)+0.05*range(xlim_curr), y0-0.17*range(ylim_curr), txt4, 'Color',[0.25 0.25 0.25], 'FontSize',8, 'Interpreter','none');

    % ---- 结果表写入：同时存 ranksum 与 t 检验 ----
    rc = rc + 1;
    grpName = ternary(whichGroup==1, "Treatment", "Control");
    resTbl(rc,:) = table( ...
        string(clf_name), grpName, ...
        sum(low),  sum(high),  med_low_all,  med_high_all,  p_all, ...
        sum(low_ex), sum(high_ex), med_low_ex, med_high_ex, p_ex, ...
        mdiff_all, p_t_all, mdiff_ex, p_t_ex, ...
        'VariableNames', {'Classifier','Group', ...
        'N_low_all','N_high_all','Median_low_all','Median_high_all','P_rank_all', ...
        'N_low_ex','N_high_ex','Median_low_ex','Median_high_ex','P_rank_ex', ...
        'MeanDiff_all','P_t_all','MeanDiff_ex','P_t_ex'} ...
        );

end

% 统一图例（NaN 句柄）
ax = gca; hold(ax, 'on');
hLowDemo  = plot(ax, NaN, NaN, 'o', 'Color', colorUse, 'MarkerFaceColor','w',  'MarkerSize',6, 'LineStyle','none');
hHighDemo = plot(ax, NaN, NaN, 'o', 'Color', colorUse, 'MarkerFaceColor',colorUse,'MarkerSize',6, 'LineStyle','none');
lg = legend([hLowDemo, hHighDemo], {'Low (score<cutoff)','High (score\ge cutoff)'}, ...
    'Orientation','horizontal','AutoUpdate','off');
try
    lg.Layout.Tile = 'south';
catch
    set(lg, 'Location','southoutside');
end
end


function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function s = num2str_or_dash(x)
if isnan(x), s = '—'; else, s = sprintf('%.3g', x); end
end


