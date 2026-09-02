function loss = simulate_from_python(rowData)
    % rowData is a 1xN double array from Python
    MRI_flag = 1;

    % Define variable names (must match your original CSV header)
    colNames = {
        'HR', 'C_SA', 'C_SV', 'C_PA', 'C_PV', 'R_SA', 'R_tSA', 'R_PA', 'R_tPA', 'R_SV', ...
        'R_t_c', 'R_p_o', 'R_p_c', 'R_m_o', 'R_m_c', 'R_a_o', 'R_a_c', ...
        'k_pas_LV', 'k_pas_RV', 'k_act_LV', 'k_act_RV', ...
        'Amref_LV', 'Amref_SEP', 'Amref_RV', ...
        'Vw_LV', 'Vw_SEP', 'Vw_RV', ...
        'RAV0u', 'LAV0u', 'LAV0c', ...
        'K1', 'Vh0', 'Height', 'Weight', 'Sex'
    };

    %try
        % Convert to table
        FakeParamsT = array2table(rowData, 'VariableNames', colNames);

        FakeParamsT.expPeri = 0.4;

        % Run simulation
        [params, init] = ProcessParamsFromVAE(FakeParamsT);
        runSimonFakeDT;
        
        o_range = struct();
        
        o_range.SBP.range = [50 300];
        o_range.DBP.range = [15 170];
        o_range.LVEDV.range = [10 1050];
        % o_range.EF.range = ;
        o_range.LVESV.range = [10 800];
        o_range.EAr.range = [0.2 8];
        o_range.LAVmax.range = [15 400];
        % o_range.LAVmin.range = ;
        % o_range.SV_LA.range = ;
        o_range.RVEDV.range = [10 700];
        o_range.RVESV.range = [10 650];
        o_range.RVEF.range = [5 85];
        % o_range.RAVmax.range = ;
        % o_range.RAVmin.range = ;
        o_range.RAPmax.range = [0 50];
        % o_range.RAPmin.range = ;
        o_range.RAPmean.range = [0 50];
        o_range.PASP.range = [0 140];
        o_range.PADP.range = [0 75];
        o_range.PCWP.range = [3 50];
        o_range.PCWPmax.range = [0 75];
        % o_range.PCWPmin.range = ;
        % o_range.CVP.range = ;
        % o_range.CVPmax.range = ;
        % o_range.CVPmin.range = ;
        o_range.CO.range = [1 15];
        o_range.Hed_LW.range = [0.2 3];
        o_range.Hed_SW.range = [0.2 3];
        o_range.Hed_RW.range = [0.2 3];
        o_range.RVEDP.range = [0 50];
        o_range.P_RV_min.range = [0 50];
        o_range.LVEDP.range = [0 100];
        o_range.P_LV_min.range = [0 50];
        o_range.RVSP.range = [5 150];
        o_range.LVESP.range = [5 250];
        o_range.MVmg.range = [0 100];
        o_range.AVpg.range = [0 150];
        % o_range.TVmg.range = ;
        o_range.PVpg.range = [0 100];
        o_range.RF_m.range = [0 100];
        o_range.RF_a.range = [0 100];
        o_range.RF_t.range = [0 100];
        o_range.RF_p.range = [0 100];
        o_range.LVIDd.range = [2 15];
        o_range.LVIDs.range = [1 10];
        % o_range.RVIDd.range = ;
        % o_range.RVIDs.range = ;
        o_range.RV_m.range = [15 400];
        o_range.LV_m.range = [35 400];
        o_range.DNA.range = [20 300];
        o_range.DNP.range = [0 100];
        o_range.MVr.range = [1 4];
        o_range.MS.range = [1 4];
        o_range.AVr.range = [1 4];
        o_range.AS.range = [1 4];
        o_range.TVr.range = [1 4];
        o_range.PVr.range = [1 4];
        o_range.PS.range = [1 4];

        % Calculate loss
        loss = 0;
        fields = fieldnames(o_range);
        
        for i = 1:length(fields)
            field = fields{i};
        
            if isfield(o_range, field) && isfield(o_range.(field), 'range') && ...
               isfield(o_vals, field) && ~isempty(o_vals.(field))
                
                % Use default scale of 0.1 unless overridden
                if isfield(o_range.(field), 'scale')
                    scale = o_range.(field).scale;
                else
                    scale = 0.1;
                end
            
                % Extract range
                range = o_range.(field).range;
            
                % Compute the penalty
                loss = loss + range_penalty(o_vals.(field), range(1), range(2), scale);
            end
        end

    % catch ME
    %     disp(['Simulation error: ' ME.message]);  % Optional: log the error
    %     loss = 1e3;  % Return large penalty on error
    % end
end


function loss = range_penalty(x, a, b, scale)
    loss = (x < a) * exp(scale * (a - x)) + (x > b) * exp(scale * (x - b));
end
