%% MLHFQ 相关性分析：9个 classifier × (治疗/对照)，时间点=1个月、1年
% 变化定义：Delta = Followup - Baseline  （负值通常代表改善）
clear; clc;

% === 读取数据 ===
T1 = readtable('CorrelationT.xlsx','VariableNamingRule','preserve','Sheet',3);
varNames = T1.Properties.VariableNames;

% 列索引
clf_cols   = 1:9;     % 9个 classifier 概率
grp_col    = 10;      % 分组：治疗=1，对照=0
ML0_col    = 13;      % MLHFQ baseline
ML1m_col   = 14;      % MLHFQ 1个月
ML1y_col   = 15;      % MLHFQ 1年

% 提取
scores = T1{:, clf_cols};
group  = T1{:, grp_col};
ML0    = T1{:, ML0_col};
ML1m   = T1{:, ML1m_col};
ML1y   = T1{:, ML1y_col};

% 基本有效性（数值非空/有限）
validML_1m = isfinite(ML0) & isfinite(ML1m);
validML_1y = isfinite(ML0) & isfinite(ML1y);

% 变化量（随访 - 基线）
dML_1m = ML1m - ML0;
dML_1y = ML1y - ML0;

% —— "无症状阈值"（可按需调整）——
mlhfq_asymp_thresh = 5;

% "始终≤阈值"标记（对应随访点）
isAlwaysLow_1m = (ML0 <= mlhfq_asymp_thresh) & (ML1m <= mlhfq_asymp_thresh) & validML_1m;
isAlwaysLow_1y = (ML0 <= mlhfq_asymp_thresh) & (ML1y <= mlhfq_asymp_thresh) & validML_1y;

% 色彩
c_treat = [0.10 0.45 0.85];   % 蓝：治疗
c_ctrl  = [0.85 0.25 0.10];   % 红：对照
alphaLevel = 0.05;            % 显著性阈值

% 结果表
res_all = table();

%% ===== 治疗组：1个月 − 基线 =====
[res1] = plot_mlhfq_3x3(scores, group, 1, ...
    dML_1m, validML_1m, isAlwaysLow_1m, ...
    clf_cols, varNames, c_treat, alphaLevel, ...
    'Treatment (label=1): MLHFQ 1m - baseline');

%% ===== 治疗组：1年 − 基线 =====
[res2] = plot_mlhfq_3x3(scores, group, 1, ...
    dML_1y, validML_1y, isAlwaysLow_1y, ...
    clf_cols, varNames, c_treat, alphaLevel, ...
    'Treatment (label=1): MLHFQ 1y - baseline');

%% ===== 对照组：1个月 − 基线 =====
[res3] = plot_mlhfq_3x3(scores, group, 0, ...
    dML_1m, validML_1m, isAlwaysLow_1m, ...
    clf_cols, varNames, c_ctrl, alphaLevel, ...
    'Control (label=0): MLHFQ 1m - baseline');

%% ===== 对照组：1年 − 基线 =====
[res4] = plot_mlhfq_3x3(scores, group, 0, ...
    dML_1y, validML_1y, isAlwaysLow_1y, ...
    clf_cols, varNames, c_ctrl, alphaLevel, ...
    'Control (label=0): MLHFQ 1y - baseline');

% 合并结果并保存
res_all = [res1; res2; res3; res4];
disp('===== MLHFQ × 9 classifiers：Spearman 结果（all vs excl. always≤thresh）=====');
disp(res_all);



%% =================== 本地函数 ===================
function resTbl = plot_mlhfq_3x3(scores, group, whichGroup, ...
    dML, validPair, isAlwaysLow, clf_cols, varNames, colorUse, alphaLevel, figTitle)

    fig = figure('Color','w','Name',figTitle);
    tlo = tiledlayout(fig,3,3,'TileSpacing','compact','Padding','compact');
    title(tlo, figTitle, 'Interpreter','none');

    resTbl = table();
    rc = 0;

    for i = 1:numel(clf_cols)
        x = scores(:, clf_cols(i));
        clf_name = varNames{clf_cols(i)};

        % 有效样本：该组 & 成对有效 & x,y非NaN
        valid = (group == whichGroup) & validPair & ~isnan(x) & ~isnan(dML);

        idx_all   = find(valid);
        idx_info  = idx_all(~isAlwaysLow(idx_all));   % 非"始终≤阈值"
        idx_noise = idx_all( isAlwaysLow(idx_all));   % "始终≤阈值"

        nexttile; hold on; box on; grid on;

        % 非"始终≤阈值"：圆点
        if ~isempty(idx_info)
            scatter(x(idx_info), dML(idx_info), 36, 'o', ...
                'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
                'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
        end
        % "始终≤阈值"：叉号
        if ~isempty(idx_noise)
            scatter(x(idx_noise), dML(idx_noise), 36, 'x', ...
                'MarkerEdgeColor', colorUse, 'LineWidth', 1.4);
        end

        xlabel(clf_name, 'Interpreter','none');
        if i==1 || i==4 || i==7
            ylabel('\Delta MLHFQ (follow-up - baseline)');
        end
        title(clf_name, 'Interpreter','none','FontWeight','normal');

        % Spearman：全部 vs 排除"始终≤阈值"
        x_all = x(valid);  y_all = dML(valid);
        [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

        keep = valid & ~isAlwaysLow;
        x_fil = x(keep);   y_fil = dML(keep);
        if numel(x_fil) >= 3
            [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
        else
            rho_fil = NaN; p_fil = NaN;
        end

        % 小注释：任一显著则红色
        % 小注释内容
        txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<125}=%.3f, p=%.3g', ...
            rho_all, p_all, rho_fil, p_fil);
        isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);
        if isSig
            txtColor = [0.85 0.15 0.15];  % 红
        else
            txtColor = [0.2 0.2 0.2];     % 灰
        end
        xlim_curr = xlim; ylim_curr = ylim;
        text(xlim_curr(1)+0.22*range(xlim_curr), ...
             ylim_curr(2)-0.65*range(ylim_curr), ...
             txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

        % 结果表写入
        rc = rc + 1;
        grpName = ternary(whichGroup==1, "Treatment", "Control");
        resTbl(rc,:) = table( ...
            string(clf_name), grpName, string(figTitle), ...
            sum(valid), rho_all, p_all, ...
            sum(keep),  rho_fil, p_fil, ...
            'VariableNames', {'Classifier','Group','Timepoint', ...
                              'N_all','Rho_all','P_all', ...
                              'N_exclLow','Rho_exclLow','P_exclLow'} ...
        );
    end

    % 统一图例
    lg = legend({'Informative (not always \le thresh)','Always \le thresh (×)'}, ...
                'Orientation','horizontal');
    lg.Layout.Tile = 'south';
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
