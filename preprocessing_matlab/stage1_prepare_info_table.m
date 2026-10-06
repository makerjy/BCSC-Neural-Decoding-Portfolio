function info_table = stage1_prepare_info_table(info_table, subject, sample_frequency)

    required_columns = { ...
        'Task', 'TMS', 'TMStarget', 'Category', ...
        'P1_TSidx', 'TrialN', 'TMS_TSidx', ...
        'Phoneme1', 'Phoneme2'};

    missing_columns = setdiff(required_columns, info_table.Properties.VariableNames);
    if ~isempty(missing_columns)
        error('Required CSV columns are missing: %s', ...
              strjoin(missing_columns, ', '));
    end

    n_rows = height(info_table);

    % Add optional timing columns when they are absent, as in P-subject CSVs.
    optional_numeric_columns = { ...
        'P2_TSidx', 'P3_TSidx'};

    for k = 1:numel(optional_numeric_columns)
        column_name = optional_numeric_columns{k};
        if ~ismember(column_name, info_table.Properties.VariableNames)
            info_table.(column_name) = nan(n_rows, 1);
        end
    end

    % Add optional label columns when absent.
    optional_text_columns = {'Stimulus', 'Phoneme3'};

    for k = 1:numel(optional_text_columns)
        column_name = optional_text_columns{k};
        if ~ismember(column_name, info_table.Properties.VariableNames)
            info_table.(column_name) = repmat("", n_rows, 1);
        end
    end

    % Convert numeric columns robustly, whether readtable imported them as
    % numbers, character arrays, cell arrays, or strings.
    numeric_columns = { ...
        'TMS', 'TrialN', 'P1_TSidx', 'P2_TSidx', ...
        'P3_TSidx', 'TMS_TSidx'};

    for k = 1:numel(numeric_columns)
        column_name = numeric_columns{k};
        values = info_table.(column_name);
        if ~isnumeric(values)
            values = str2double(string(values));
        end
        info_table.(column_name) = double(values);
    end

    % Preserve the original CSV columns and add standardized columns for
    % event generation. This handles single-phoneme vs single_phoneme and
    % Words vs words consistently.
    task = lower(strtrim(string(info_table.Task)));
    task(ismissing(task)) = "";
    task = replace(task, "-", "_");
    task = replace(task, " ", "_");
    task(task == "singlephoneme") = "single_phoneme";

    stimulus_target = lower(strtrim(string(info_table.TMStarget)));
    stimulus_target(ismissing(stimulus_target)) = "";
    stimulus_target = replace(stimulus_target, "-", "_");
    stimulus_target = replace(stimulus_target, " ", "_");

    category = lower(strtrim(string(info_table.Category)));
    category(ismissing(category)) = "";
    category = replace(category, "-", "_");
    category = replace(category, " ", "_");

    tms = info_table.TMS;

    control_rows = (tms == 0);
    stimulus_target(control_rows) = "control";
    condition = repmat("control", n_rows, 1);
    condition(tms == 1) = "tms";
    % stimulation_site(stimulus_target == "control") = "none";

    info_table.TaskCanonical = task;
    info_table.TMSCanonical = tms;
    info_table.ConditionCanonical = condition;
    info_table.TargetCanonical = stimulus_target;
    % info_table.StimulationSite = stimulation_site;
    info_table.CategoryCanonical = category;
    info_table.csv_row = (1:n_rows)';
    info_table.Subject = repmat(string(subject), n_rows, 1);

    % Recreate timing anchors from P1. For control trials, tms_1 and tms_2
    % are nominal time anchors, not actual magnetic pulses.
    info_table.Base_Line = ...
        info_table.P1_TSidx - round(0.5 * sample_frequency);
    info_table.Onset_TSidx = ...
        info_table.P1_TSidx - round(0.2 * sample_frequency);
    info_table.TMS1_TSidx = ...
        info_table.P1_TSidx - round(0.1 * sample_frequency);
    info_table.TMS2_TSidx = ...
        info_table.P1_TSidx - round(0.05 * sample_frequency);
    info_table.Final_TSidx = ...
        info_table.P1_TSidx + round(1.0 * sample_frequency);

    % The CSV TMS_TSidx is the second pulse time for phonemes/Words. Use it
    % as a consistency check. single-phoneme has no actual TMS, so its value
    % is not used for this check.
    timing_check_rows = ...
        task ~= "single_phoneme" & isfinite(info_table.TMS_TSidx);

    if any(timing_check_rows)
        timing_error = abs( ...
            info_table.TMS_TSidx(timing_check_rows) - ...
            info_table.TMS2_TSidx(timing_check_rows));

        if any(timing_error > 1)
            error([ ...
                'CSV timing indices do not match EEG.srate = %.3f Hz. ' ...
                'The selected .set may already be resampled or may not ' ...
                'belong to this CSV. Maximum mismatch: %.3f samples.'], ...
                sample_frequency, max(timing_error));
        end
    end
end