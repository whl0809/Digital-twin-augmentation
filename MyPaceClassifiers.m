%% ======= Master: BNP / MLHFQ / Activity / AF burden 相关性 + (BNP阈值分组检验) =======
% 说明：
% - 读表方式与第一个脚本完全一致：Sheet = subset + 1
% - 公共变量与函数共享；不保存任何文件，仅屏显/画图
% - 三块内容：
%    (A) NT-proBNP：相关性（治疗/对照 3x3）+ 基于训练cutoff的 High/Low 分组比较（fractional change）
%    (B) MLHFQ：相关性（治疗/对照 × 1m/1y，3x3）
%    (C) Activity & AF burden：相关性（治疗/对照 × 6m/1y，3x3）

clear; clc;
SHOW_CORR_PLOTS = false;   % 相关性图是否显示；false=全部不画

%% ========= 统一读表 =========
subset = 2;  % 1 for 6k  2 for 25K
T1 = readtable('CorrelationT.xlsx','VariableNamingRule','preserve','Sheet',subset+1);
CutoffT = readtable("optimal_thresholds_summary.csv");
if subset == 1
    Traincutoff = CutoffT.subset_threshold;
else
    Traincutoff = CutoffT.full_threshold;
end
varNames = T1.Properties.VariableNames;

% 统一列索引
clf_cols   = 1:9;     % 9个 classifier 概率
grp_col    = 10;      % 分组：治疗=1，对照=0

% BNP
BNP0_col   = 11;      % baseline NT-proBNP
BNP1m_col  = 12;      % 1个月 NT-proBNP

% MLHFQ
ML0_col    = 13;      % MLHFQ baseline
ML1m_col   = 14;      % MLHFQ 1个月
ML1y_col   = 15;      % MLHFQ 1年

% Activity
ACT0_col   = 16;      % baseline
ACT6m_col  = 17;      % 6个月
ACT1y_col  = 18;      % 1年

% AF burden（%）
AF0_col    = 19;      % baseline
AF6m_col   = 20;      % 6个月
AF1y_col   = 21;      % 1年

% 统一提取
scores = T1{:, clf_cols};
group  = T1{:, grp_col};

BNP0  = T1{:, BNP0_col};   BNP1m = T1{:, BNP1m_col};
ML0   = T1{:, ML0_col};    ML1m  = T1{:, ML1m_col};    ML1y = T1{:, ML1y_col};
ACT0  = T1{:, ACT0_col};   ACT6m = T1{:, ACT6m_col};   ACT1y = T1{:, ACT1y_col};
AF0   = T1{:, AF0_col};    AF6m  = T1{:, AF6m_col};    AF1y  = T1{:, AF1y_col};

% 颜色
c_treat = [0.10 0.45 0.85];
c_ctrl  = [0.85 0.25 0.10];
alphaLevel = 0.05;

%% ========================= (A) NT-proBNP =========================
% —— 基本校验 ——
validBNP = BNP0 > 0 & BNP1m > 0 & isfinite(BNP0) & isfinite(BNP1m);
if any(~validBNP)
    warning('发现 %d 个 BNP<=0 或非有限值样本，将在分析中忽略这些样本。', sum(~validBNP));
end

% ΔBNP（负值=改善）与 fractional change（与 myPACE 口径一致）
dBNP = BNP1m - BNP0;
dFracBNP = (BNP1m - BNP0) ./ BNP0;   % (1m - baseline)/baseline
yLabelBNP_frac = 'Fractional change in NT-proBNP ((1m - baseline) / baseline)';

% "始终正常"：基线与1月都 <125 pg/mL
isAlwaysNormal = (BNP0 < 125) & (BNP1m < 125);

% 结果表
res_BNP = table(); rowCounter = 0;
if SHOW_CORR_PLOTS
    % —— 治疗组（3×3）——
    fig1 = figure('Color','w','Name','Treatment (1)');
    tlo1 = tiledlayout(fig1,3,3,'TileSpacing','compact','Padding','compact');
    title(tlo1,'Treatment group (label=1): Classifier vs \Delta NT-proBNP');

    for i = 1:numel(clf_cols)
        clf_name = varNames{clf_cols(i)};
        x = scores(:, i);

        valid = validBNP & group==1 & ~isnan(x) & ~isnan(dBNP);
        idx_all   = find(valid);
        idx_info  = idx_all(~isAlwaysNormal(idx_all));
        idx_noise = idx_all( isAlwaysNormal(idx_all));

        nexttile; hold on; box on; grid on;

        if ~isempty(idx_info)
            scatter(x(idx_info), dBNP(idx_info), 36, 'o', ...
                'MarkerEdgeColor', c_treat, 'MarkerFaceColor', c_treat, ...
                'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
        end
        if ~isempty(idx_noise)
            scatter(x(idx_noise), dBNP(idx_noise), 36, 'x', ...
                'MarkerEdgeColor', c_treat, 'LineWidth', 1.4);
        end

        xlabel(clf_name, 'Interpreter','none');
        if i==1 || i==4 || i==7
            ylabel('\Delta NT-proBNP (baseline - 1m)');
        end
        title(clf_name, 'Interpreter','none','FontWeight','normal');

        x_all = x(valid); y_all = dBNP(valid);
        [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

        keep = valid & ~isAlwaysNormal;
        x_fil = x(keep); y_fil = dBNP(keep);
        if numel(x_fil) >= 3
            [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
        else
            rho_fil = NaN; p_fil = NaN;
        end

        isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);
        txtColor = ternary(isSig, [0.85 0.15 0.15], [0.2 0.2 0.2]);
        txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<=th}=%.3f, p=%.3g', ...
            rho_all, p_all, rho_fil, p_fil);

        xlim_curr = xlim; ylim_curr = ylim;
        text(xlim_curr(1)+0.22*range(xlim_curr), ylim_curr(2)-0.65*range(ylim_curr), ...
            txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

        rowCounter = rowCounter + 1;
        res_BNP(rowCounter,:) = table( ...
            string(clf_name), "Treatment", ...
            sum(valid), rho_all, p_all, ...
            sum(keep),  rho_fil, p_fil, ...
            'VariableNames', {'Classifier','Group','N_all','Rho_all','P_all','N_excl125','Rho_excl125','P_excl125'} ...
            );
    end
    lg1 = legend({'Informative (not both <125)','Always <125 (×)'}, 'Orientation','horizontal');
    lg1.Layout.Tile = 'south';

    % —— 对照组（3×3）——
    fig0 = figure('Color','w','Name','Control (0)');
    tlo0 = tiledlayout(fig0,3,3,'TileSpacing','compact','Padding','compact');
    title(tlo0,'Control group (label=0): Classifier vs \Delta NT-proBNP');

    for i = 1:numel(clf_cols)
        clf_name = varNames{clf_cols(i)}; x = scores(:, i);

        valid = validBNP & group==0 & ~isnan(x) & ~isnan(dBNP);
        idx_all   = find(valid);
        idx_info  = idx_all(~isAlwaysNormal(idx_all));
        idx_noise = idx_all( isAlwaysNormal(idx_all));

        nexttile; hold on; box on; grid on;

        if ~isempty(idx_info)
            scatter(x(idx_info), dBNP(idx_info), 36, 'o', ...
                'MarkerEdgeColor', c_ctrl, 'MarkerFaceColor', c_ctrl, ...
                'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
        end
        if ~isempty(idx_noise)
            scatter(x(idx_noise), dBNP(idx_noise), 36, 'x', ...
                'MarkerEdgeColor', c_ctrl, 'LineWidth', 1.4);
        end

        xlabel(clf_name, 'Interpreter','none');
        if i==1 || i==4 || i==7
            ylabel('\Delta NT-proBNP (baseline - 1m)');
        end
        title(clf_name, 'Interpreter','none','FontWeight','normal');

        x_all = x(valid); y_all = dBNP(valid);
        [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

        keep = valid & ~isAlwaysNormal;
        x_fil = x(keep); y_fil = dBNP(keep);
        if numel(x_fil) >= 3
            [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
        else
            rho_fil = NaN; p_fil = NaN;
        end

        isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);
        txtColor = ternary(isSig, [0.85 0.15 0.15], [0.2 0.2 0.2]);
        txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<=th}=%.3f, p=%.3g', ...
            rho_all, p_all, rho_fil, p_fil);

        xlim_curr = xlim; ylim_curr = ylim;
        text(xlim_curr(1)+0.22*range(xlim_curr), ...
            ylim_curr(2)-0.65*range(ylim_curr), ...
            txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

        rowCounter = rowCounter + 1;
        res_BNP(rowCounter,:) = table( ...
            string(clf_name), "Control", ...
            sum(valid), rho_all, p_all, ...
            sum(keep),  rho_fil, p_fil, ...
            'VariableNames', {'Classifier','Group','N_all','Rho_all','P_all','N_excl125','Rho_excl125','P_excl125'} ...
            );
    end
    lg0 = legend({'Informative (not both <125)','Always <125 (×)'}, 'Orientation','horizontal');
    lg0.Layout.Tile = 'south';


    disp('===== BNP × 9 classifiers：Spearman 结果（all vs excl. <125）=====');
    disp(res_BNP);
end


%% —— 基于训练cutoff，把9个分数二分为 High(1)/Low(0)，并在治疗/对照分别比较 ΔNT-proBNP（fractional）——
cutvec = resolve_cutvec_positional(Traincutoff, numel(clf_cols));
HL = bsxfun(@ge, scores, cutvec);   % N×9 logical

resHL_T = compare_high_low_3x3(HL, group, 1, dFracBNP, validBNP, isAlwaysNormal, ...
    clf_cols, varNames, yLabelBNP_frac, ...
    'Treatment (label=1): High vs Low (fixed cutoff; fractional change)', ...
    c_treat, alphaLevel);

resHL_C = compare_high_low_3x3(HL, group, 0, dFracBNP, validBNP, isAlwaysNormal, ...
    clf_cols, varNames, yLabelBNP_frac, ...
    'Control (label=0): High vs Low (fixed cutoff; fractional change)', ...
    c_ctrl, alphaLevel);

disp('===== High vs Low (fixed training cutoff) on ΔNT-proBNP (fractional) =====');
disp([resHL_T; resHL_C]);


%% ========================= (B) MLHFQ =========================
% 有效性
validML_1m = isfinite(ML0) & isfinite(ML1m);
validML_1y = isfinite(ML0) & isfinite(ML1y);

if any(~validML_1m)
    warning('发现 %d 个 MLHFQ1m<=0 或非有限值样本，将在分析中忽略这些样本。', sum(~validML_1m));
end
if any(~validML_1y)
    warning('发现 %d 个 MLHFQ1y<=0 或非有限值样本，将在分析中忽略这些样本。', sum(~validML_1y));
end
% 变化量（负值通常代表改善）
dML_1m = ML1m - ML0;
dML_1y = ML1y - ML0;

% "无症状阈值"与标记
mlhfq_asymp_thresh = -1;
isAlwaysLow_1m = (ML0 <= mlhfq_asymp_thresh) & (ML1m <= mlhfq_asymp_thresh) & validML_1m;
isAlwaysLow_1y = (ML0 <= mlhfq_asymp_thresh) & (ML1y <= mlhfq_asymp_thresh) & validML_1y;

% 治疗/对照 × (1m / 1y)
if SHOW_CORR_PLOTS
    resML_T_1m = plot_mlhfq_3x3(scores, group, 1, dML_1m, validML_1m, isAlwaysLow_1m, ...
        clf_cols, varNames, c_treat, alphaLevel, 'Treatment (label=1): MLHFQ 1m - baseline');

    resML_T_1y = plot_mlhfq_3x3(scores, group, 1, dML_1y, validML_1y, isAlwaysLow_1y, ...
        clf_cols, varNames, c_treat, alphaLevel, 'Treatment (label=1): MLHFQ 1y - baseline');

    resML_C_1m = plot_mlhfq_3x3(scores, group, 0, dML_1m, validML_1m, isAlwaysLow_1m, ...
        clf_cols, varNames, c_ctrl, alphaLevel, 'Control (label=0): MLHFQ 1m - baseline');

    resML_C_1y = plot_mlhfq_3x3(scores, group, 0, dML_1y, validML_1y, isAlwaysLow_1y, ...
        clf_cols, varNames, c_ctrl, alphaLevel, 'Control (label=0): MLHFQ 1y - baseline');

    res_MLHFQ_all = [resML_T_1m; resML_T_1y; resML_C_1m; resML_C_1y];
    disp('===== MLHFQ × 9 classifiers：Spearman 结果（all vs excl. always≤thresh）=====');
    disp(res_MLHFQ_all);
end


%% ===== MLHFQ：固定训练 cutoff 的 High/Low 比较（Δ）=====
yLabelML = '\Delta MLHFQ (follow-up - baseline)';

% 1个月随访
resHL_ML_T_1m = compare_high_low_3x3(HL, group, 1, ...
    dML_1m, validML_1m, isAlwaysLow_1m, ...
    clf_cols, varNames, yLabelML, ...
    'Treatment (label=1): High vs Low (fixed cutoff) on \Delta MLHFQ (1m - baseline)', ...
    c_treat, alphaLevel);

resHL_ML_C_1m = compare_high_low_3x3(HL, group, 0, ...
    dML_1m, validML_1m, isAlwaysLow_1m, ...
    clf_cols, varNames, yLabelML, ...
    'Control (label=0): High vs Low (fixed cutoff) on \Delta MLHFQ (1m - baseline)', ...
    c_ctrl, alphaLevel);

% 1年随访
resHL_ML_T_1y = compare_high_low_3x3(HL, group, 1, ...
    dML_1y, validML_1y, isAlwaysLow_1y, ...
    clf_cols, varNames, yLabelML, ...
    'Treatment (label=1): High vs Low (fixed cutoff) on \Delta MLHFQ (1y - baseline)', ...
    c_treat, alphaLevel);

resHL_ML_C_1y = compare_high_low_3x3(HL, group, 0, ...
    dML_1y, validML_1y, isAlwaysLow_1y, ...
    clf_cols, varNames, yLabelML, ...
    'Control (label=0): High vs Low (fixed cutoff) on \Delta MLHFQ (1y - baseline)', ...
    c_ctrl, alphaLevel);

disp('===== High vs Low (fixed training cutoff) on \Delta MLHFQ =====');
disp([resHL_ML_T_1m; resHL_ML_C_1m; resHL_ML_T_1y; resHL_ML_C_1y]);

%% ========================= (C) Activity =========================
% 有效性
vACT_6m = isfinite(ACT0) & isfinite(ACT6m);
vACT_1y = isfinite(ACT0) & isfinite(ACT1y);
vAF_6m  = isfinite(AF0)  & isfinite(AF6m);
vAF_1y  = isfinite(AF0)  & isfinite(AF1y);

% 变化量
dACT_6m = ACT6m - ACT0;  % 正值=改善
dACT_1y = ACT1y - ACT0;

dAF_6m  = AF6m  - AF0;   % 负值=改善
dAF_1y  = AF1y  - AF0;

% 阈值（与原脚本一致）
act_high_thresh = NaN;     % 禁用"始终高"逻辑
af_low_thresh   = 1;       % %

% "始终阈值侧"标记
isAlwaysHigh_ACT_6m = vACT_6m & isfinite(act_high_thresh) & (ACT0 >= act_high_thresh) & (ACT6m >= act_high_thresh);
isAlwaysHigh_ACT_1y = vACT_1y & isfinite(act_high_thresh) & (ACT0 >= act_high_thresh) & (ACT1y >= act_high_thresh);

isAlwaysLow_AF_6m   = vAF_6m  & (AF0 <= af_low_thresh) & (AF6m <= af_low_thresh);
isAlwaysLow_AF_1y   = vAF_1y  & (AF0 <= af_low_thresh) & (AF1y <= af_low_thresh);
if SHOW_CORR_PLOTS
    % Activity：治疗/对照 × (6m/1y)
    res_ACT_T_6m = plot_delta_3x3(scores, group, 1, dACT_6m, vACT_6m, isAlwaysHigh_ACT_6m, ...
        clf_cols, varNames, c_treat, alphaLevel, ...
        'Treatment (label=1): Activity 6m - baseline', ...
        '\Delta Activity (follow-up - baseline)', ...
        'Always >= thresh (×)');

    res_ACT_T_1y = plot_delta_3x3(scores, group, 1, dACT_1y, vACT_1y, isAlwaysHigh_ACT_1y, ...
        clf_cols, varNames, c_treat, alphaLevel, ...
        'Treatment (label=1): Activity 1y - baseline', ...
        '\Delta Activity (follow-up - baseline)', ...
        'Always >= thresh (×)');

    res_ACT_C_6m = plot_delta_3x3(scores, group, 0, dACT_6m, vACT_6m, isAlwaysHigh_ACT_6m, ...
        clf_cols, varNames, c_ctrl, alphaLevel, ...
        'Control (label=0): Activity 6m - baseline', ...
        '\Delta Activity (follow-up - baseline)', ...
        'Always >= thresh (×)');

    res_ACT_C_1y = plot_delta_3x3(scores, group, 0, dACT_1y, vACT_1y, isAlwaysHigh_ACT_1y, ...
        clf_cols, varNames, c_ctrl, alphaLevel, ...
        'Control (label=0): Activity 1y - baseline', ...
        '\Delta Activity (follow-up - baseline)', ...
        'Always >= thresh (×)');

    res_ACT_all = [res_ACT_T_6m; res_ACT_T_1y; res_ACT_C_6m; res_ACT_C_1y];
    disp('===== Activity × 9 classifiers：Spearman 结果（all vs excl. always-high）=====');
    disp(res_ACT_all);
end

%% ===== Activity：固定训练 cutoff 的 High/Low 比较（Δ）=====
yLabelACT = '\Delta Activity (follow-up - baseline)';

% 6个月随访
resHL_ACT_T_6m = compare_high_low_3x3(HL, group, 1, ...
    dACT_6m, vACT_6m, isAlwaysHigh_ACT_6m, ...
    clf_cols, varNames, yLabelACT, ...
    'Treatment (label=1): High vs Low (fixed cutoff) on \Delta Activity (6m - baseline)', ...
    c_treat, alphaLevel);

resHL_ACT_C_6m = compare_high_low_3x3(HL, group, 0, ...
    dACT_6m, vACT_6m, isAlwaysHigh_ACT_6m, ...
    clf_cols, varNames, yLabelACT, ...
    'Control (label=0): High vs Low (fixed cutoff) on \Delta Activity (6m - baseline)', ...
    c_ctrl, alphaLevel);

% 1年随访
resHL_ACT_T_1y = compare_high_low_3x3(HL, group, 1, ...
    dACT_1y, vACT_1y, isAlwaysHigh_ACT_1y, ...
    clf_cols, varNames, yLabelACT, ...
    'Treatment (label=1): High vs Low (fixed cutoff) on \Delta Activity (1y - baseline)', ...
    c_treat, alphaLevel);

resHL_ACT_C_1y = compare_high_low_3x3(HL, group, 0, ...
    dACT_1y, vACT_1y, isAlwaysHigh_ACT_1y, ...
    clf_cols, varNames, yLabelACT, ...
    'Control (label=0): High vs Low (fixed cutoff) on \Delta Activity (1y - baseline)', ...
    c_ctrl, alphaLevel);

disp('===== High vs Low (fixed training cutoff) on \Delta Activity =====');
disp([resHL_ACT_T_6m; resHL_ACT_C_6m; resHL_ACT_T_1y; resHL_ACT_C_1y]);

%% ========================= (D) AF Burden =========================
% AF burden：治疗/对照 × (6m/1y)
if SHOW_CORR_PLOTS
    res_AF_T_6m = plot_delta_3x3(scores, group, 1, dAF_6m, vAF_6m, isAlwaysLow_AF_6m, ...
        clf_cols, varNames, c_treat, alphaLevel, ...
        'Treatment (label=1): AF burden 6m - baseline', ...
        '\Delta AF burden (% follow-up - % baseline)', ...
        'Always <= thresh (×)');

    res_AF_T_1y = plot_delta_3x3(scores, group, 1, dAF_1y, vAF_1y, isAlwaysLow_AF_1y, ...
        clf_cols, varNames, c_treat, alphaLevel, ...
        'Treatment (label=1): AF burden 1y - baseline', ...
        '\Delta AF burden (% follow-up - % baseline)', ...
        'Always <= thresh (×)');

    res_AF_C_6m = plot_delta_3x3(scores, group, 0, dAF_6m, vAF_6m, isAlwaysLow_AF_6m, ...
        clf_cols, varNames, c_ctrl, alphaLevel, ...
        'Control (label=0): AF burden 6m - baseline', ...
        '\Delta AF burden (% follow-up - % baseline)', ...
        'Always <= thresh (×)');

    res_AF_C_1y = plot_delta_3x3(scores, group, 0, dAF_1y, vAF_1y, isAlwaysLow_AF_1y, ...
        clf_cols, varNames, c_ctrl, alphaLevel, ...
        'Control (label=0): AF burden 1y - baseline', ...
        '\Delta AF burden (% follow-up - % baseline)', ...
        'Always <= thresh (×)');

    res_AF_all = [res_AF_T_6m; res_AF_T_1y; res_AF_C_6m; res_AF_C_1y];
    disp('===== AF burden × 9 classifiers：Spearman 结果（all vs excl. always-low）=====');
    disp(res_AF_all);
end


%% =================== 公共/本地函数（原样口径） ===================
function cutvec = resolve_cutvec_positional(Traincutoff, M)
try
    numcols = varfun(@isnumeric, Traincutoff, 'OutputFormat','uniform');
    data = Traincutoff{:, numcols};
catch
    data = Traincutoff(:, :);
end
if ~isnumeric(data)
    error('Traincutoff 表含非数值列；请确保阈值是数值，并把非数值列去掉或放在后面。');
end
vals = data(:)';
if numel(vals) < M
    error('阈值数量不足：需要 %d 个，实际只有 %d 个。', M, numel(vals));
end
cutvec = vals(1:M);
end

function resTbl = compare_high_low_3x3(HL, group, whichGroup, ...
    dY, validPair, isAlwaysTrivial, clf_cols, varNames, yLabel, figTitle, colorUse, alphaLevel)

    % 始终画分组比较图（你要求保留）
    fig = figure('Color','w','Name',figTitle);
    tlo = tiledlayout(fig,3,3,'TileSpacing','compact','Padding','compact');
    title(tlo, figTitle, 'Interpreter','none');

    resTbl = table();
    rc = 0;

    for k = 1:numel(clf_cols)
        clf_name = varNames{clf_cols(k)};

        valid  = (group==whichGroup) & validPair & ~isnan(dY);
        high   = HL(:, k) & valid;
        low    = ~HL(:, k) & valid;

        keep_ex = valid & ~isAlwaysTrivial;
        high_ex = HL(:, k) & keep_ex;
        low_ex  = ~HL(:, k) & keep_ex;

        nexttile; hold on; box on; grid on;

        % 箱线 + 抖点（保留你原风格）
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

        % —— 仅 Welch t 检验（all / excl）——
        p_t_all = NaN;  mdiff_all = NaN;
        p_t_ex  = NaN;  mdiff_ex  = NaN;

        if sum(low)>=2 && sum(high)>=2
            [~, p_t_all] = ttest2(dY(high), dY(low), 'Vartype','unequal');  % High vs Low
            mdiff_all = mean(dY(high),'omitnan') - mean(dY(low),'omitnan'); % H-L
        end
        if sum(low_ex)>=2 && sum(high_ex)>=2
            [~, p_t_ex] = ttest2(dY(high_ex), dY(low_ex), 'Vartype','unequal');
            mdiff_ex = mean(dY(high_ex),'omitnan') - mean(dY(low_ex),'omitnan');
        end

        % 注释（仅 pt 与均值差）
        isSig = ( ~isnan(p_t_all) && p_t_all < alphaLevel ) || ...
                ( ~isnan(p_t_ex)  && p_t_ex  < alphaLevel );
        txtColor = isSig*[0.85 0.15 0.15] + (~isSig)*[0.2 0.2 0.2];

        txt1 = sprintf('N_a_l_l L=%d,H=%d | pt=%.3g', sum(low), sum(high), p_t_all);
        txt2 = sprintf('N_e_x  L=%d,H=%d | pt=%.3g', sum(low_ex), sum(high_ex), p_t_ex);
     

        xlim_curr = xlim; ylim_curr = ylim;
        y0 = ylim_curr(2)-0.08*range(ylim_curr);
        text(xlim_curr(1)+0.05*range(xlim_curr), y0,                       txt1, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');
        text(xlim_curr(1)+0.05*range(xlim_curr), y0-0.06*range(ylim_curr), txt2, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

        % —— 返回表：仅 t 检验结果 —— 
        rc = rc + 1;
        grpName = ternary(whichGroup==1, "Treatment", "Control");
        resTbl(rc,:) = table( ...
            string(clf_name), grpName, ...
            sum(low),  sum(high),  mdiff_all,  p_t_all, ...
            sum(low_ex), sum(high_ex), mdiff_ex, p_t_ex, ...
            'VariableNames', {'Classifier','Group', ...
            'N_low_all','N_high_all','MeanDiff_all','P_t_all', ...
            'N_low_ex','N_high_ex','MeanDiff_ex','P_t_ex'} ...
        );
    end

    % 统一图例（Low/High）
    ax = gca; hold(ax, 'on');
    hLowDemo  = plot(ax, NaN, NaN, 'o', 'Color', colorUse, 'MarkerFaceColor','w',  'MarkerSize',6, 'LineStyle','none');
    hHighDemo = plot(ax, NaN, NaN, 'o', 'Color', colorUse, 'MarkerFaceColor',colorUse,'MarkerSize',6, 'LineStyle','none');
    lg = legend([hLowDemo, hHighDemo], {'Low (score<cutoff)','High (score\ge cutoff)'}, ...
        'Orientation','horizontal','AutoUpdate','off');
    try
        lg.Layout.Tile = 'south'; 
    catch
        set(lg,'Location','southoutside'); 
    end
end


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

    valid = (group == whichGroup) & validPair & ~isnan(x) & ~isnan(dML);

    idx_all   = find(valid);
    idx_info  = idx_all(~isAlwaysLow(idx_all));
    idx_noise = idx_all( isAlwaysLow(idx_all));

    nexttile; hold on; box on; grid on;

    if ~isempty(idx_info)
        scatter(x(idx_info), dML(idx_info), 36, 'o', ...
            'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
            'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
    end
    if ~isempty(idx_noise)
        scatter(x(idx_noise), dML(idx_noise), 36, 'x', ...
            'MarkerEdgeColor', colorUse, 'LineWidth', 1.4);
    end

    xlabel(clf_name, 'Interpreter','none');
    if i==1 || i==4 || i==7
        ylabel('\Delta MLHFQ (follow-up - baseline)');
    end
    title(clf_name, 'Interpreter','none','FontWeight','normal');

    x_all = x(valid);  y_all = dML(valid);
    [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

    keep = valid & ~isAlwaysLow;
    x_fil = x(keep);   y_fil = dML(keep);
    if numel(x_fil) >= 3
        [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
    else
        rho_fil = NaN; p_fil = NaN;
    end

    txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<=th}=%.3f, p=%.3g', ...
        rho_all, p_all, rho_fil, p_fil);
    isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);
    txtColor = ternary(isSig, [0.85 0.15 0.15], [0.2 0.2 0.2]);
    xlim_curr = xlim; ylim_curr = ylim;
    text(xlim_curr(1)+0.22*range(xlim_curr), ...
        ylim_curr(2)-0.65*range(ylim_curr), ...
        txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

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

lg = legend({'Informative (not always \le thresh)','Always \le thresh (×)'}, ...
    'Orientation','horizontal');
lg.Layout.Tile = 'south';
end

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

    valid = (group == whichGroup) & validPair & ~isnan(x) & ~isnan(deltaY);

    idx_all   = find(valid);
    idx_info  = idx_all(~isAlwaysTrivial(idx_all));
    idx_noise = idx_all( isAlwaysTrivial(idx_all));

    nexttile; hold on; box on; grid on;

    if ~isempty(idx_info)
        scatter(x(idx_info), deltaY(idx_info), 36, 'o', ...
            'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
            'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.85);
    end
    if ~isempty(idx_noise)
        scatter(x(idx_noise), deltaY(idx_noise), 36, 'x', ...
            'MarkerEdgeColor', colorUse, 'LineWidth', 1.4);
    end

    xlabel(clf_name, 'Interpreter','none');
    if i==1 || i==4 || i==7
        ylabel(yLabel, 'Interpreter','none');
    end
    title(clf_name, 'Interpreter','none','FontWeight','normal');

    x_all = x(valid);   y_all = deltaY(valid);
    [rho_all, p_all] = corr(x_all, y_all, 'Type','Spearman','Rows','complete');

    keep = valid & ~isAlwaysTrivial;
    x_fil = x(keep);    y_fil = deltaY(keep);
    if numel(x_fil) >= 3
        [rho_fil, p_fil] = corr(x_fil, y_fil, 'Type','Spearman','Rows','complete');
    else
        rho_fil = NaN; p_fil = NaN;
    end

    txt = sprintf('\\rho_{all}=%.3f, p=%.3g\n\\rho_{ex<=th}=%.3f, p=%.3g', ...
        rho_all, p_all, rho_fil, p_fil);
    isSig = (p_all < alphaLevel) || (p_fil < alphaLevel);
    txtColor = isSig * [0.85 0.15 0.15] + (~isSig) * [0.2 0.2 0.2];

    xlim_curr = xlim; ylim_curr = ylim;
    text(xlim_curr(1)+0.22*range(xlim_curr), ...
        ylim_curr(2)-0.65*range(ylim_curr), ...
        txt, 'Color', txtColor, 'FontSize',9, 'Interpreter','tex');

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

ax = gca; hold(ax, 'on');
hInfoDemo = plot(ax, NaN, NaN, 'o', ...
    'MarkerEdgeColor', colorUse, 'MarkerFaceColor', colorUse, ...
    'MarkerSize', 6, 'LineStyle','none');
hTrivDemo = plot(ax, NaN, NaN, 'x', ...
    'Color', colorUse, 'LineWidth', 1.4, 'MarkerSize', 6, 'LineStyle','none');
lg = legend([hInfoDemo, hTrivDemo], ...
    {'Informative (not always threshold-side)', trivialLegend}, ...
    'Orientation','horizontal', 'AutoUpdate','off');
lg.Layout.Tile = 'south';
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
