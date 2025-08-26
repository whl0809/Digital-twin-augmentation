%% ========= Activity & AF burden 相关性：9个classifier × (治疗/对照) =========
% 变化定义：Delta = Followup - Baseline
% - Activity：正值=改善
% - AF burden：负值=改善
clear; clc;

% === 读数据 ===
T1 = readtable('CorrelationT.xlsx','VariableNamingRule','preserve','Sheet',3);
varNames = T1.Properties.VariableNames;

% 列索引
clf_cols   = 1:9;    % 9个classifier概率
grp_col    = 10;     % 分组：治疗=1，对照=0

% Activity 列
ACT0_col   = 16;     % baseline
ACT6m_col  = 17;     % 6个月
ACT1y_col  = 18;     % 1年

% AF burden 列（单位：百分比 %）
AF0_col    = 19;     % baseline
AF6m_col   = 20;     % 6个月
AF1y_col   = 21;     % 1年

% 提取
scores = T1{:, clf_cols};
group  = T1{:, grp_col};

ACT0   = T1{:, ACT0_col};   ACT6m  = T1{:, ACT6m_col};   ACT1y  = T1{:, ACT1y_col};
AF0    = T1{:, AF0_col};    AF6m   = T1{:, AF6m_col};    AF1y   = T1{:, AF1y_col};

% 有效性（有限数）
vACT_6m = isfinite(ACT0) & isfinite(ACT6m);
vACT_1y = isfinite(ACT0) & isfinite(ACT1y);
vAF_6m  = isfinite(AF0)  & isfinite(AF6m);
vAF_1y  = isfinite(AF0)  & isfinite(AF1y);

% 变化量
dACT_6m = ACT6m - ACT0;
dACT_1y = ACT1y - ACT0;

dAF_6m  = AF6m  - AF0;   % 负值=改善（AF占比下降）
dAF_1y  = AF1y  - AF0;

% 阈值（可改）
act_high_thresh = NaN;   % 默认禁用；若启用：基线与随访均 >= 阈值 视为"始终高"
af_low_thresh   = 1;     % %：基线与随访均 <= 1% 视为"始终低"

% 标记"始终在阈值一侧"（用于图上叉号 + 在过滤版Spearman里剔除）
isAlwaysHigh_ACT_6m = vACT_6m & isfinite(act_high_thresh) & ...
    (ACT0 >= act_high_thresh) & (ACT6m >= act_high_thresh);
isAlwaysHigh_ACT_1y = vACT_1y & isfinite(act_high_thresh) & ...
    (ACT0 >= act_high_thresh) & (ACT1y >= act_high_thresh);

isAlwaysLow_AF_6m   = vAF_6m  & (AF0 <= af_low_thresh) & (AF6m <= af_low_thresh);
isAlwaysLow_AF_1y   = vAF_1y  & (AF0 <= af_low_thresh) & (AF1y <= af_low_thresh);

% 颜色 & 显著性阈值
c_treat = [0.10 0.45 0.85];   % 蓝：治疗
c_ctrl  = [0.85 0.25 0.10];   % 红：对照
alphaLevel = 0.05;

%% ========================= Activity =========================
% --- 治疗组：6m-baseline ---
res_ACT_T_6m = plot_delta_3x3(scores, group, 1, ...
    dACT_6m, vACT_6m, isAlwaysHigh_ACT_6m, ...
    clf_cols, varNames, c_treat, alphaLevel, ...
    'Treatment (label=1): Activity 6m - baseline', ...
    '\Delta Activity (follow-up - baseline)', ...
    'Always >= thresh (×)');

% --- 治疗组：1y-baseline ---
res_ACT_T_1y = plot_delta_3x3(scores, group, 1, ...
    dACT_1y, vACT_1y, isAlwaysHigh_ACT_1y, ...
    clf_cols, varNames, c_treat, alphaLevel, ...
    'Treatment (label=1): Activity 1y - baseline', ...
    '\Delta Activity (follow-up - baseline)', ...
    'Always >= thresh (×)');

% --- 对照组：6m-baseline ---
res_ACT_C_6m = plot_delta_3x3(scores, group, 0, ...
    dACT_6m, vACT_6m, isAlwaysHigh_ACT_6m, ...
    clf_cols, varNames, c_ctrl, alphaLevel, ...
    'Control (label=0): Activity 6m - baseline', ...
    '\Delta Activity (follow-up - baseline)', ...
    'Always >= thresh (×)');

% --- 对照组：1y-baseline ---
res_ACT_C_1y = plot_delta_3x3(scores, group, 0, ...
    dACT_1y, vACT_1y, isAlwaysHigh_ACT_1y, ...
    clf_cols, varNames, c_ctrl, alphaLevel, ...
    'Control (label=0): Activity 1y - baseline', ...
    '\Delta Activity (follow-up - baseline)', ...
    'Always >= thresh (×)');

res_ACT_all = [res_ACT_T_6m; res_ACT_T_1y; res_ACT_C_6m; res_ACT_C_1y];
disp('===== Activity × 9 classifiers：Spearman 结果（all vs excl. always-high）=====');
disp(res_ACT_all);


%% ========================= AF burden =========================
% --- 治疗组：6m-baseline ---
res_AF_T_6m = plot_delta_3x3(scores, group, 1, ...
    dAF_6m, vAF_6m, isAlwaysLow_AF_6m, ...
    clf_cols, varNames, c_treat, alphaLevel, ...
    'Treatment (label=1): AF burden 6m - baseline', ...
    '\Delta AF burden (% follow-up - % baseline)', ...
    'Always <= thresh (×)');

% --- 治疗组：1y-baseline ---
res_AF_T_1y = plot_delta_3x3(scores, group, 1, ...
    dAF_1y, vAF_1y, isAlwaysLow_AF_1y, ...
    clf_cols, varNames, c_treat, alphaLevel, ...
    'Treatment (label=1): AF burden 1y - baseline', ...
    '\Delta AF burden (% follow-up - % baseline)', ...
    'Always <= thresh (×)');

% --- 对照组：6m-baseline ---
res_AF_C_6m = plot_delta_3x3(scores, group, 0, ...
    dAF_6m, vAF_6m, isAlwaysLow_AF_6m, ...
    clf_cols, varNames, c_ctrl, alphaLevel, ...
    'Control (label=0): AF burden 6m - baseline', ...
    '\Delta AF burden (% follow-up - % baseline)', ...
    'Always <= thresh (×)');

% --- 对照组：1y-baseline ---
res_AF_C_1y = plot_delta_3x3(scores, group, 0, ...
    dAF_1y, vAF_1y, isAlwaysLow_AF_1y, ...
    clf_cols, varNames, c_ctrl, alphaLevel, ...
    'Control (label=0): AF burden 1y - baseline', ...
    '\Delta AF burden (% follow-up - % baseline)', ...
    'Always <= thresh (×)');

res_AF_all = [res_AF_T_6m; res_AF_T_1y; res_AF_C_6m; res_AF_C_1y];
disp('===== AF burden × 9 classifiers：Spearman 结果（all vs excl. always-low）=====');
disp(res_AF_all);


%% =================== 通用绘图函数 ===================
function resTbl = plot_delta_3x3(scores, group, whichGroup, ...
    deltaY, validPair, isAlwaysTrivial, clf_cols, varNames, colorUse, ...
    alphaLevel, figTitle, yLabel, trivialLegend)

fig = figure('Color','w','Name',figTitle);
tlo = tiledlayout(fig,3,3,'TileSpacing','compact','Padding','compact');
title(tlo, figTitle, 'Interpreter','none');

resTbl = table();
rc = 0;

for i = 1:numel(clf_cols)
    x = scores(:, clf_cols(i));
    clf_name = varNames{clf_cols(i)};

    % 有效样本：该组 & 成对有效 & x,y非NaN
    valid = (group == whichGroup) & validPair & ~isnan(x) & ~isnan(deltaY);

    idx_all   = find(valid);
    idx_info  = idx_all(~isAlwaysTrivial(idx_all));  % 非"始终阈值侧"
    idx_noise = idx_all( isAlwaysTrivial(idx_all));  % "始终阈值侧"

    nexttile; hold on; box on; grid on;

    % 非"始终阈值侧"：圆点
    if ~isempty(idx_info)
        scatter(x(idx_info), deltaY(idx_info), 36, 'o', ...
            'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
            'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
    end
    % "始终阈值侧"：叉号
    if ~isempty(idx_noise)
        scatter(x(idx_noise), deltaY(idx_noise), 36, 'x', ...
            'MarkerEdgeColor', colorUse, 'LineWidth', 1.4);
    end

    xlabel(clf_name, 'Interpreter','none');
    if i==1 || i==4 || i==7
        ylabel(yLabel, 'Interpreter','none');
    end
    title(clf_name, 'Interpreter','none','FontWeight','normal');

    % Spearman：全部 vs 排除"始终阈值侧"
    x_all = x(valid);   y_all = deltaY(valid);
    [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

    keep = valid & ~isAlwaysTrivial;
    x_fil = x(keep);    y_fil = deltaY(keep);
    if numel(x_fil) >= 3
        [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
    else
        rho_fil = NaN; p_fil = NaN;
    end

    % 小注释（任一显著则红）
    txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<=/>=th}=%.3f, p=%.3g', ...
        rho_all, p_all, rho_fil, p_fil);
    isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);
    txtColor = isSig * [0.85 0.15 0.15] + (~isSig) * [0.2 0.2 0.2];

    xlim_curr = xlim; ylim_curr = ylim;
    text(xlim_curr(1)+0.22*range(xlim_curr), ...
        ylim_curr(2)-0.65*range(ylim_curr), ...
        txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

    % 结果表
    rc = rc + 1;
    grpName = ternary(whichGroup==1, "Treatment", "Control");
    resTbl(rc,:) = table( ...
        string(clf_name), grpName, string(figTitle), ...
        sum(valid), rho_all, p_all, ...
        sum(keep),  rho_fil, p_fil, ...
        'VariableNames', {'Classifier','Group','Panel', ...
        'N_all','Rho_all','P_all', ...
        'N_exclTriv','Rho_exclTriv','P_exclTriv'} ...
        );
end

% —— 统一图例（兼容旧版）：用 NaN 句柄保证始终有两项 —— 
% 在当前轴（最后一个子图）上造两个"示例句柄"，不会出现在图中，但能进图例
ax = gca; hold(ax, 'on');
hInfoDemo = plot(ax, NaN, NaN, 'o', ...
    'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
    'MarkerSize', 6, 'LineStyle','none');
hTrivDemo = plot(ax, NaN, NaN, 'x', ...
    'Color', colorUse, 'LineWidth', 1.4, 'MarkerSize', 6, 'LineStyle','none');

% 创建图例（不要把 tlo 传给 legend）
lg = legend([hInfoDemo, hTrivDemo], ...
    {'Informative (not always threshold-side)', trivialLegend}, ...
    'Orientation','horizontal', 'AutoUpdate','off');

% 把图例放到底部（tiledlayout 的 south）
% 注：不用将 tlo 传入 legend；直接设置 Layout.Tile 属性即可
lg.Layout.Tile = 'south';


end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
