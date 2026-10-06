function export_stage7_to_mat(root_folder, mat_output_folder)
%EXPORT_STAGE7_TO_MAT Export stage-7 EEGLAB datasets to simple MAT files.
%
% This function uses EEGLAB/MATLAB itself to read the custom .set files.
% It therefore avoids the MNE/scipy EEGLAB-event parsing error.
%
% Usage:
%   export_stage7_to_mat( ...
%       fullfile(project_root, 'preprocessing_test'), ...
%       fullfile(project_root, 'stage7_mat_export'));
%
% Output:
%   One *_stage7_export.mat file per subject.
%
% The following Python step converts these MAT files to the exact requested
% NPZ schema:
%   python convert_stage7_mat_to_npz.py --input ... --output ...

    if nargin ~= 2
        error([ ...
            'Usage: export_stage7_to_mat(root_folder, ' ...
            'mat_output_folder)']);
    end

    if isstring(root_folder)
        root_folder = char(root_folder);
    end

    if isstring(mat_output_folder)
        mat_output_folder = char(mat_output_folder);
    end

    if ~ischar(root_folder) || ~ischar(mat_output_folder)
        error('Both input arguments must be folder paths.');
    end

    if exist('pop_loadset', 'file') ~= 2
        error([ ...
            'EEGLAB is not on the MATLAB path. Start EEGLAB first or add ' ...
            'the EEGLAB folder to the MATLAB path.']);
    end

    if ~isfolder(root_folder)
        error('Root folder does not exist: %s', root_folder);
    end

    if ~isfolder(mat_output_folder)
        mkdir(mat_output_folder);
    end

    files = dir(fullfile( ...
        root_folder, ...
        '**', ...
        '*_7_ica_components_removed.set'));

    if isempty(files)
        error('No *_7_ica_components_removed.set files found under: %s', ...
            root_folder);
    end

    fprintf('Found %d stage-7 file(s).\n', numel(files));

    summary_subject = strings(numel(files), 1);
    summary_status = strings(numel(files), 1);
    summary_input = strings(numel(files), 1);
    summary_output = strings(numel(files), 1);
    summary_message = strings(numel(files), 1);
    summary_trials = nan(numel(files), 1);
    summary_channels = nan(numel(files), 1);
    summary_times = nan(numel(files), 1);

    for file_index = 1:numel(files)

        set_path = fullfile(files(file_index).folder, files(file_index).name);
        [~, set_stem, ~] = fileparts(set_path);

        token = regexp( ...
            files(file_index).name, ...
            '^([PS]\d+)_7_ica_components_removed\.set$', ...
            'tokens', ...
            'once', ...
            'ignorecase');

        if isempty(token)
            fprintf('[SKIP] Unexpected filename: %s\n', set_path);
            continue;
        end

        subject_id = upper(string(token{1}));
        output_path = fullfile( ...
            mat_output_folder, ...
            char(subject_id + "_stage7_export.mat"));

        summary_subject(file_index) = subject_id;
        summary_input(file_index) = string(set_path);
        summary_output(file_index) = string(output_path);

        fprintf('\n[%s] Loading:\n    %s\n', char(subject_id), set_path);

        try
            EEG = pop_loadset( ...
                'filename', files(file_index).name, ...
                'filepath', files(file_index).folder);

            EEG = eeg_checkset(EEG);
            EEG = eeg_checkset(EEG, 'eventconsistency');

            if EEG.trials <= 1
                error('Stage-7 dataset is not epoched: EEG.trials=%d', ...
                    EEG.trials);
            end

            if any(~isfinite(EEG.data(:)))
                error('EEG.data contains NaN or Inf.');
            end

            n_trials = double(EEG.trials);
            n_channels = double(EEG.nbchan);
            n_times = double(EEG.pnts);

            % Required signal layout: trials x channels x timepoints.
            X = permute(single(EEG.data), [3 1 2]);

            times = double(EEG.times(:)) / 1000;
            sfreq = double(EEG.srate);

            channel_names_string = string({EEG.chanlocs.labels})';
            channel_names = cellstr(channel_names_string);

            channel_types_string = repmat("eeg", n_channels, 1);

            if isfield(EEG.chanlocs, 'type')
                for channel_index = 1:n_channels
                    current_type = local_text( ...
                        EEG.chanlocs(channel_index), ...
                        {'type'}, ...
                        "eeg");

                    if strlength(current_type) > 0
                        channel_types_string(channel_index) = lower(current_type);
                    end
                end
            end

            channel_types = cellstr(channel_types_string);

            % Raw EEGLAB coordinates are exported only as an intermediate.
            % The Python step maps standard channel names to MNE's
            % standard_1005 montage in metres.
            channel_positions_xyz_raw = nan(n_channels, 3);

            for channel_index = 1:n_channels
                channel_positions_xyz_raw(channel_index, 1) = ...
                    local_numeric( ...
                        EEG.chanlocs(channel_index), {'X'}, NaN);
                channel_positions_xyz_raw(channel_index, 2) = ...
                    local_numeric( ...
                        EEG.chanlocs(channel_index), {'Y'}, NaN);
                channel_positions_xyz_raw(channel_index, 3) = ...
                    local_numeric( ...
                        EEG.chanlocs(channel_index), {'Z'}, NaN);
            end

            subject_string = repmat(subject_id, n_trials, 1);
            task_string = strings(n_trials, 1);
            tms_condition_string = strings(n_trials, 1);
            tms_target_string = strings(n_trials, 1);
            stimulation_site_string = strings(n_trials, 1);
            category_string = strings(n_trials, 1);

            phoneme1_string = strings(n_trials, 1);
            phoneme2_string = strings(n_trials, 1);
            phoneme3_string = strings(n_trials, 1);
            stimulus_string = strings(n_trials, 1);

            csv_row = -ones(n_trials, 1, 'int32');
            trial_n = -ones(n_trials, 1, 'int32');
            original_epoch = zeros(n_trials, 1, 'int32');
            current_epoch = int32((1:n_trials)');

            stim1_time = nan(n_trials, 1);
            stim2_time = nan(n_trials, 1);
            stim3_time = nan(n_trials, 1);
            tms1_time = nan(n_trials, 1);
            tms2_time = nan(n_trials, 1);

            tms_applied = false(n_trials, 1);

            event_epoch_number = nan(numel(EEG.event), 1);

            for event_index = 1:numel(EEG.event)
                event_epoch_number(event_index) = local_numeric( ...
                    EEG.event(event_index), {'epoch'}, NaN);
            end

            for epoch_index = 1:n_trials

                current_event_indices = find( ...
                    event_epoch_number == epoch_index);

                if isempty(current_event_indices)
                    error('Epoch %d contains no EEG.event entries.', ...
                        epoch_index);
                end

                current_roles = strings(numel(current_event_indices), 1);
                current_times = nan(numel(current_event_indices), 1);

                for local_index = 1:numel(current_event_indices)
                    event_index = current_event_indices(local_index);

                    current_roles(local_index) = ...
                        local_event_role(EEG.event(event_index));

                    current_times(local_index) = ...
                        local_event_relative_time( ...
                            EEG.event(event_index), ...
                            epoch_index, ...
                            n_times, ...
                            sfreq, ...
                            double(EEG.xmin));
                end

                stim1_local_indices = find(current_roles == "stim_1");

                if isempty(stim1_local_indices)
                    error('Epoch %d contains no stim_1 event.', epoch_index);
                end

                [~, closest_index] = min( ...
                    abs(current_times(stim1_local_indices)));

                anchor_local_index = ...
                    stim1_local_indices(closest_index);

                anchor_event_index = ...
                    current_event_indices(anchor_local_index);

                anchor_event = EEG.event(anchor_event_index);

                task_string(epoch_index) = ...
                    local_text(anchor_event, {'task'}, "");

                tms_target_string(epoch_index) = ...
                    local_text( ...
                        anchor_event, ...
                        {'tms_target', 'target'}, ...
                        "");

                stimulation_site_string(epoch_index) = ...
                    local_text( ...
                        anchor_event, ...
                        {'stimulation_site'}, ...
                        "");

                category_string(epoch_index) = ...
                    local_text(anchor_event, {'category'}, "");

                phoneme1_string(epoch_index) = ...
                    local_text(anchor_event, {'phoneme1'}, "");

                phoneme2_string(epoch_index) = ...
                    local_text(anchor_event, {'phoneme2'}, "");

                phoneme3_string(epoch_index) = ...
                    local_text(anchor_event, {'phoneme3'}, "");

                stimulus_string(epoch_index) = ...
                    local_text(anchor_event, {'stimulus'}, "");

                csv_row(epoch_index) = int32(local_numeric( ...
                    anchor_event, {'csv_row'}, -1));

                trial_n(epoch_index) = int32(local_numeric( ...
                    anchor_event, {'trial_number', 'trial_n'}, -1));

                tms_applied(epoch_index) = ...
                    local_tms_applied(anchor_event);

                if tms_applied(epoch_index)
                    tms_condition_string(epoch_index) = "tms";
                else
                    tms_condition_string(epoch_index) = "control";
                end

                if epoch_index <= numel(EEG.epoch) && ...
                        isfield(EEG.epoch, 'original_epoch')

                    original_epoch(epoch_index) = int32(local_numeric( ...
                        EEG.epoch(epoch_index), ...
                        {'original_epoch'}, ...
                        epoch_index));
                else
                    original_epoch(epoch_index) = int32(epoch_index);
                end

                stim1_time(epoch_index) = local_role_time( ...
                    current_roles, current_times, "stim_1", true);

                stim2_time(epoch_index) = local_role_time( ...
                    current_roles, current_times, "stim_2", true);

                stim3_time(epoch_index) = local_role_time( ...
                    current_roles, current_times, "stim_3", true);

                tms1_time(epoch_index) = local_role_time( ...
                    current_roles, current_times, "tms_1", false);

                tms2_time(epoch_index) = local_role_time( ...
                    current_roles, current_times, "tms_2", false);

                if ~isfinite(stim1_time(epoch_index))
                    stim1_time(epoch_index) = 0;
                end

                % The current stage-7 epoch begins at stim_1, so these
                % events normally no longer exist inside EEG.event.
                if ~isfinite(tms1_time(epoch_index))
                    tms1_time(epoch_index) = -0.100;
                end

                if ~isfinite(tms2_time(epoch_index))
                    tms2_time(epoch_index) = -0.050;
                end
            end

            if any(csv_row < 1)
                bad_epochs = find(csv_row < 1);
                n_bad_to_show = min(numel(bad_epochs), 20);
                error( ...
                    'Missing csv_row in current epoch(s): %s', ...
                    mat2str(bad_epochs(1:n_bad_to_show)'));
            end

            if numel(unique(csv_row)) ~= n_trials
                error('csv_row is not unique across stage-7 epochs.');
            end

            tolerance = 0.5 / sfreq;
            epoch_tmin = times(1);
            epoch_tmax = times(end);

            stim1_in_epoch = local_in_epoch( ...
                stim1_time, epoch_tmin, epoch_tmax, tolerance);
            stim2_in_epoch = local_in_epoch( ...
                stim2_time, epoch_tmin, epoch_tmax, tolerance);
            stim3_in_epoch = local_in_epoch( ...
                stim3_time, epoch_tmin, epoch_tmax, tolerance);
            tms1_in_epoch = local_in_epoch( ...
                tms1_time, epoch_tmin, epoch_tmax, tolerance);
            tms2_in_epoch = local_in_epoch( ...
                tms2_time, epoch_tmin, epoch_tmax, tolerance);

            % Save text as cell arrays of char for reliable scipy.io.loadmat.
            subject = cellstr(subject_string);
            task = cellstr(task_string);
            tms_condition = cellstr(tms_condition_string);
            tms_target = cellstr(tms_target_string);
            stimulation_site = cellstr(stimulation_site_string);
            category = cellstr(category_string);
            phoneme1 = cellstr(phoneme1_string);
            phoneme2 = cellstr(phoneme2_string);
            phoneme3 = cellstr(phoneme3_string);
            stimulus = cellstr(stimulus_string);
            source_set = cellstr(repmat(string(set_path), n_trials, 1));

            save( ...
                output_path, ...
                'X', ...
                'times', ...
                'sfreq', ...
                'channel_names', ...
                'channel_types', ...
                'channel_positions_xyz_raw', ...
                'subject', ...
                'task', ...
                'tms_condition', ...
                'tms_target', ...
                'stimulation_site', ...
                'category', ...
                'phoneme1', ...
                'phoneme2', ...
                'phoneme3', ...
                'stimulus', ...
                'csv_row', ...
                'trial_n', ...
                'original_epoch', ...
                'current_epoch', ...
                'source_set', ...
                'stim1_time', ...
                'stim2_time', ...
                'stim3_time', ...
                'tms1_time', ...
                'tms2_time', ...
                'tms_applied', ...
                'stim1_in_epoch', ...
                'stim2_in_epoch', ...
                'stim3_in_epoch', ...
                'tms1_in_epoch', ...
                'tms2_in_epoch', ...
                '-v7');

            fprintf( ...
                '[%s] EXPORTED: X=(%d, %d, %d)\n    %s\n', ...
                char(subject_id), ...
                n_trials, ...
                n_channels, ...
                n_times, ...
                output_path);

            summary_status(file_index) = "EXPORTED";
            summary_message(file_index) = "OK";
            summary_trials(file_index) = n_trials;
            summary_channels(file_index) = n_channels;
            summary_times(file_index) = n_times;

        catch ME
            fprintf( ...
                2, ...
                '[%s] FAILED: %s\n', ...
                char(subject_id), ...
                ME.message);

            summary_status(file_index) = "FAILED";
            summary_message(file_index) = string(getReport( ...
                ME, 'extended', 'hyperlinks', 'off'));
        end
    end

    summary_table = table( ...
        summary_subject, ...
        summary_status, ...
        summary_input, ...
        summary_output, ...
        summary_trials, ...
        summary_channels, ...
        summary_times, ...
        summary_message, ...
        'VariableNames', { ...
            'Subject', ...
            'Status', ...
            'InputSet', ...
            'OutputMat', ...
            'NTrials', ...
            'NChannels', ...
            'NTimes', ...
            'Message'});

    summary_path = fullfile( ...
        mat_output_folder, ...
        'stage7_mat_export_summary.csv');

    writetable(summary_table, summary_path);

    fprintf('\nSummary saved:\n%s\n', summary_path);
end


function role = local_event_role(event_record)

    role = local_text(event_record, {'event_role'}, "");

    if strlength(role) > 0
        role = lower(strtrim(role));
        return;
    end

    event_type = local_text(event_record, {'type'}, "");
    event_type = lower(strtrim(event_type));

    pieces = split(event_type, "/");

    if isempty(pieces)
        role = event_type;
    else
        role = pieces(end);
    end
end


function relative_time = local_event_relative_time( ...
        event_record, epoch_index, pnts, sfreq, xmin)

    latency = local_numeric(event_record, {'latency'}, NaN);

    if ~isfinite(latency)
        relative_time = NaN;
        return;
    end

    first_sample_latency = (epoch_index - 1) * pnts + 1;
    sample_offset = latency - first_sample_latency;

    relative_time = xmin + sample_offset / sfreq;
end


function output = local_role_time( ...
        roles, relative_times, requested_role, prefer_nonnegative)

    matching = find(roles == requested_role);

    if isempty(matching)
        output = NaN;
        return;
    end

    values = relative_times(matching);
    values = values(isfinite(values));

    if isempty(values)
        output = NaN;
        return;
    end

    if prefer_nonnegative
        nonnegative = sort(values(values >= -1e-9));

        if ~isempty(nonnegative)
            output = nonnegative(1);
            return;
        end
    end

    [~, nearest_index] = min(abs(values));
    output = values(nearest_index);
end


function applied = local_tms_applied(event_record)

    numeric_tms = local_numeric(event_record, {'tms'}, NaN);

    if isfinite(numeric_tms)
        applied = numeric_tms == 1;
        return;
    end

    condition = lower(local_text( ...
        event_record, ...
        {'tms_condition', 'condition'}, ...
        ""));

    if any(condition == ["control", "tms0", "false", "0"])
        applied = false;
        return;
    end

    if any(condition == ["tms", "tms1", "true", "1"])
        applied = true;
        return;
    end

    target = lower(local_text( ...
        event_record, ...
        {'tms_target', 'target'}, ...
        ""));

    applied = strlength(target) > 0 && target ~= "control";
end


function output = local_text(record, candidate_fields, default_value)

    output = string(default_value);

    for field_index = 1:numel(candidate_fields)
        field_name = candidate_fields{field_index};

        if ~isfield(record, field_name)
            continue;
        end

        value = record.(field_name);

        if isempty(value)
            continue;
        end

        if iscell(value)
            value = value{1};
        end

        if isstring(value)
            text = value(1);
        elseif ischar(value)
            text = string(value);
        elseif isnumeric(value) || islogical(value)
            text = string(value(1));
        else
            try
                text = string(value);
                text = text(1);
            catch
                continue;
            end
        end

        text = strtrim(text);

        if ismissing(text) || ...
                strlength(text) == 0 || ...
                lower(text) == "nan" || ...
                lower(text) == "<missing>"
            continue;
        end

        output = text;
        return;
    end
end


function output = local_numeric(record, candidate_fields, default_value)

    output = double(default_value);

    for field_index = 1:numel(candidate_fields)
        field_name = candidate_fields{field_index};

        if ~isfield(record, field_name)
            continue;
        end

        value = record.(field_name);

        if isempty(value)
            continue;
        end

        if iscell(value)
            value = value{1};
        end

        if isnumeric(value) || islogical(value)
            candidate = double(value(1));
        else
            candidate = str2double(string(value(1)));
        end

        if isfinite(candidate)
            output = candidate;
            return;
        end
    end
end


function inside = local_in_epoch(values, tmin, tmax, tolerance)

    inside = ...
        isfinite(values) & ...
        values >= (tmin - tolerance) & ...
        values <= (tmax + tolerance);
end
