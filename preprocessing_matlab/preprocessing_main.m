% EEG DATA ANALYSIS
% Rebuilt event-setting section for all tasks and conditions

%% SET THE ENVIRONMENT

clear;
clc;

project_root = fileparts(fileparts(mfilename('fullpath')));
csv_start_folder = fullfile(project_root, 'data', 'events_information');
[table_file_name, table_file_path] = uigetfile( ...
    fullfile(csv_start_folder, '*.csv'), ...
    'Select the table with the events');

info_table_file = fullfile(table_file_path, table_file_name);
[~, table_base_name, ~] = fileparts(table_file_name);
subject_table = stage1_extract_subject_id(table_base_name);

fprintf('Info Table File: %s\n', info_table_file);
fprintf('CSV Subject: %s\n', subject_table);

set_start_folder = fullfile(project_root, 'data', 'raw_data');
[file_name, path_name] = uigetfile(fullfile(set_start_folder, '*.set'), 'Select recording');

[~, subject_set, ~] = fileparts(file_name);
fprintf('.set Subject: %s\n', subject_set);

if ~strcmpi(subject_table, subject_set)
    error(['CSV subject (%s) and .set subject (%s) do not match. ' ...
           'Select files from the same subject.'], ...
           subject_table, subject_set);
end

subject = upper(subject_table);

[ALLEEG, ~, ~, ALLCOM] = eeglab;
EEG = pop_loadset('filename', file_name, 'filepath', path_name);
EEG = eeg_checkset(EEG);

save_folder = fullfile(pwd, subject);
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end
fprintf('Save Folder: %s\n', save_folder);

% REMOVE AUXILIARY CHANNELS WHEN PRESENT

% These channels are not scalp EEG channels.
% Some subjects already have them removed, while others still contain
% some of the BIP channels.

channels_to_exclude = {'EOG', 'M1', 'M2'};

for bip_index = 1:24
    channels_to_exclude{end + 1} = sprintf('BIP%d', bip_index);
end

current_channel_labels = ...
    {EEG.chanlocs.labels};

% Normalize labels so that variations such as BIP03, BIP_3 and BIP-3
% can also be identified.

normalized_current_labels = cellfun( ...
    @stage1_normalize_auxiliary_channel_label, ...
    current_channel_labels, ...
    'UniformOutput', false);

normalized_exclusion_labels = cellfun( ...
    @stage1_normalize_auxiliary_channel_label, ...
    channels_to_exclude, ...
    'UniformOutput', false);

channels_to_remove = find( ...
    ismember( ...
        normalized_current_labels, ...
        normalized_exclusion_labels));

fprintf('\n=== Auxiliary-channel removal ===\n');
fprintf('Channels before removal: %d\n', EEG.nbchan);

if isempty(channels_to_remove)

    fprintf('No EOG, M1, M2, or BIP1-BIP24 channels were present.\n');

else

    fprintf('Removing %d auxiliary channel(s):\n', ...
        numel(channels_to_remove));

    for channel_index = channels_to_remove

        fprintf('  %s\n', ...
            current_channel_labels{channel_index});
    end

    EEG = pop_select( ...
        EEG, ...
        'nochannel', ...
        channels_to_remove);

    EEG = eeg_checkset(EEG);
end

fprintf('Channels after removal: %d\n', EEG.nbchan);

%% ADD CHANNEL LOCATIONS ONLY WHEN MISSING

channel_has_valid_location = false(1, EEG.nbchan);

for channel_index = 1:EEG.nbchan

    current_location = EEG.chanlocs(channel_index);

    valid_xyz = ...
        isfield(current_location, 'X') && ...
        isnumeric(current_location.X) && ...
        isscalar(current_location.X) && ...
        isfinite(current_location.X) && ...
        isfield(current_location, 'Y') && ...
        isnumeric(current_location.Y) && ...
        isscalar(current_location.Y) && ...
        isfinite(current_location.Y) && ...
        isfield(current_location, 'Z') && ...
        isnumeric(current_location.Z) && ...
        isscalar(current_location.Z) && ...
        isfinite(current_location.Z);

    valid_polar = ...
        isfield(current_location, 'theta') && ...
        isnumeric(current_location.theta) && ...
        isscalar(current_location.theta) && ...
        isfinite(current_location.theta) && ...
        isfield(current_location, 'radius') && ...
        isnumeric(current_location.radius) && ...
        isscalar(current_location.radius) && ...
        isfinite(current_location.radius);

    channel_has_valid_location(channel_index) = ...
        valid_xyz || valid_polar;
end


if all(channel_has_valid_location)

    fprintf('\n=== Channel locations ===\n');
    fprintf('All %d channels already have location information.\n', EEG.nbchan);
    fprintf('S01 channel-location copying was skipped.\n');

else

    missing_location_indices = ...
        find(~channel_has_valid_location);

    fprintf('\n=== Channel locations ===\n');
    fprintf('Channels missing location information: %d / %d\n', ...
        numel(missing_location_indices), ...
        EEG.nbchan);

    fprintf('Loading channel locations from raw S01.set...\n');


    % LOAD REFERENCE S01 ONLY WHEN NEEDED

    % Channel-location reference dataset
    reference_s01_file = fullfile( ...
        project_root, 'data', 'raw_data', 'S01.set');
    
    if ~exist(reference_s01_file, 'file')
        error('Reference S01 dataset was not found: %s', ...
            reference_s01_file);
    end

    [reference_s01_path, ...
     reference_s01_name, ...
     reference_s01_extension] = ...
        fileparts(reference_s01_file);

    reference_EEG = pop_loadset( ...
        'filename', ...
        [reference_s01_name reference_s01_extension], ...
        'filepath', ...
        reference_s01_path, ...
        'loadmode', ...
        'info');

    reference_EEG = eeg_checkset(reference_EEG);

    current_channel_labels = ...
        strtrim(string({EEG.chanlocs.labels}));

    reference_channel_labels = ...
        strtrim(string({reference_EEG.chanlocs.labels}));


   % COPY ONLY MISSING CHANNEL LOCATIONS

    coordinate_fields = { ...
        'theta', ...
        'radius', ...
        'X', ...
        'Y', ...
        'Z', ...
        'sph_theta', ...
        'sph_phi', ...
        'sph_radius'};

    for channel_index = missing_location_indices

        current_label = ...
            current_channel_labels(channel_index);

        matching_reference_index = find( ...
            strcmpi( ...
                reference_channel_labels, ...
                current_label), ...
            1);

        if isempty(matching_reference_index)

            error('Channel "%s" was not found in raw S01.set.', ...
                current_label);
        end

        for field_index = 1:numel(coordinate_fields)

            field_name = ...
                coordinate_fields{field_index};

            if isfield(reference_EEG.chanlocs, field_name)

                EEG.chanlocs(channel_index).(field_name) = ...
                    reference_EEG.chanlocs( ...
                        matching_reference_index).(field_name);
            end
        end

        fprintf('Copied location: %s\n', current_label);
    end

    EEG = eeg_checkset(EEG);

end

% FINAL CHANNEL-LOCATION VALIDATION

invalid_location_indices = [];

for channel_index = 1:EEG.nbchan

    current_location = EEG.chanlocs(channel_index);

    valid_xyz = ...
        isnumeric(current_location.X) && ...
        isscalar(current_location.X) && ...
        isfinite(current_location.X) && ...
        isnumeric(current_location.Y) && ...
        isscalar(current_location.Y) && ...
        isfinite(current_location.Y) && ...
        isnumeric(current_location.Z) && ...
        isscalar(current_location.Z) && ...
        isfinite(current_location.Z);

    valid_polar = ...
        isnumeric(current_location.theta) && ...
        isscalar(current_location.theta) && ...
        isfinite(current_location.theta) && ...
        isnumeric(current_location.radius) && ...
        isscalar(current_location.radius) && ...
        isfinite(current_location.radius);

    if ~(valid_xyz || valid_polar)

        invalid_location_indices(end + 1) = ...
            channel_index;
    end
end

if ~isempty(invalid_location_indices)

    fprintf('\nChannels still missing valid locations:\n');

    for channel_index = invalid_location_indices

        fprintf('  %s\n', ...
            EEG.chanlocs(channel_index).labels);
    end

    error('Some EEG channels still have no valid location information.');
end

fprintf('Validated channel locations: %d / %d\n', ...
    EEG.nbchan, EEG.nbchan);

% Store the dataset only after auxiliary-channel removal.

[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);

eeglab redraw;

%% READ, NORMALIZE, AND VALIDATE THE TRIAL TABLE

sample_frequency = EEG.srate;

opts = detectImportOptions(info_table_file);

text_columns = { ...
    'Stimulus', ...
    'TMStarget', ...
    'Task', ...
    'Phoneme1', ...
    'Phoneme2', ...
    'Phoneme3', ...
    'Category'};

existing_text_columns = intersect( ...
    text_columns, ...
    opts.VariableNames, ...
    'stable');

opts = setvartype(opts, existing_text_columns, 'string');

info_table = readtable(info_table_file, opts);

for k = 1:numel(existing_text_columns)
    column_name = existing_text_columns{k};
    values = info_table.(column_name);
    values(ismissing(values)) = "";
    info_table.(column_name) = values;
end

% function: prepare_info_table
info_table = stage1_prepare_info_table(info_table, subject, sample_frequency);
% function: validate_info_table
stage1_validate_info_table(info_table, subject, EEG);

fprintf('\nSampling frequency: %.3f Hz\n', sample_frequency);
fprintf('Number of CSV rows: %d\n', height(info_table));

% TrialN is not necessarily unique in the P-subject tables. csv_row is the
% unique table-row identifier that should be used for later alignment.
if numel(unique(info_table.TrialN)) < height(info_table)
    warning(['TrialN is repeated in this CSV. Use EEG.event.csv_row or ' ...
             'EEG.event.latency, not TrialN alone, to align rejected trials.']);
end

% function: make_trial_summary
summary_table = stage1_make_trial_summary(info_table);
fprintf('\n=== Trial summary ===\n');
disp(summary_table);

%% SET ALL EVENTS

% The source .set can already contain generic events such as "TMS" and
% "stimulus". They are archived in EEG.etc and replaced so that rerunning
% this section does not duplicate events.

% function: create_all_events
EEG = stage1_create_all_events(EEG, info_table, info_table_file, summary_table);

fprintf('Created %d structured events.\n', numel(EEG.event));
fprintf(['Event label format:\n' ...
         '  task/tms0-or-tms1/target/category/event_role\n']);
fprintf(['Example:\n' ...
         '  phonemes/tms1/tongue/alveolar/stim_1\n']);

%% 1. SAVE DATASET

save_name = fullfile(save_folder, [subject '_1_set_events.set']);

[ALLEEG, EEG, CURRENTSET] = pop_newset( ...
    ALLEEG, EEG, CURRENTSET, ...
    'setname', 'Set events', ...
    'savenew', save_name, ...
    'gui', 'off');

fprintf('Saved: %s\n', save_name);
eeglab redraw;

%% BAD TMS SEGMENTS INTERPOLATION

% pre-epoching, continuous EEG data

if EEG.trials ~= 1
    error([ ...
        'TMS interpolation must be performed before epoching. ' ...
        'The current EEG dataset contains %d trials.'], ...
        EEG.trials);
end

tms_true_rows = find(info_table.TMSCanonical == 1);

tms_true_index = [info_table.TMS1_TSidx(tms_true_rows); ...
                  info_table.TMS2_TSidx(tms_true_rows)];

tms_true_index = unique(round(tms_true_index), 'sorted');

% check range
pre_tms_samples  = round(0.005 * sample_frequency);
post_tms_samples = round(0.025 * sample_frequency);
first_bad_samples = tms_true_index - pre_tms_samples;
last_bad_samples  = tms_true_index + post_tms_samples;
total_samples = EEG.pnts;
invalid_pulses = ...
    first_bad_samples <= 1 | ...
    last_bad_samples >= total_samples;
if any(invalid_pulses)
    error([ ...
        'A TMS interpolation interval is outside the EEG recording. ' ...
        'Problematic TMS sample index or indices: %s'], ...
        mat2str(tms_true_index(invalid_pulses)'));
end

% TMS artifact range: -5 ms ~ +25 ms
for index = tms_true_index'                                                                 
    tms_interval = (index - pre_tms_samples):(index + post_tms_samples);     
    EEG.data(:,tms_interval) = nan;                                                    
end

fprintf('\n=== TMS artifact interpolation ===\n');
fprintf('Number of actual TMS trials: %d\n', numel(tms_true_rows));
fprintf('Number of TMS pulses: %d\n', numel(tms_true_index));
fprintf('Interpolation window: -5 ms to +25 ms\n');

for ch = 1:EEG.nbchan
    disp("Bad TMS ringing/step artifect removed from channel " + string(ch));
    EEG.data(ch,:) = fillgaps(double(EEG.data(ch,:)), round(0.25*sample_frequency), 25);
end

n_interpolated_pulses = numel(tms_true_index);

fprintf('Finished interpolating %d TMS pulses.\n', ...
    n_interpolated_pulses);

EEG = eeg_checkset(EEG);

%% 2. SAVE TMS INTERPOLATED DATASET

save_name = fullfile(save_folder, [subject '_2_tms_interpolated.set']);

[ALLEEG, EEG, CURRENTSET] = pop_newset( ...
    ALLEEG, EEG, CURRENTSET, ...
    'setname', 'TMS artifact interpolated', ...
    'savenew', save_name, ...
    'gui', 'off');

fprintf('Saved: %s\n', save_name);
fprintf('Interpolated TMS pulses: %d\n', n_interpolated_pulses);

eeglab redraw;

%% RESAMPLE, NOTCH FILTER AND BANDPASS FILTER

% Paper-based ERP preprocessing:
%   1. Resample continuous EEG to 256 Hz
%   2. Apply 59-61 Hz notch filter
%   3. Apply 0.1-100 Hz band-pass filter
%
% This section must be performed before epoching.

if EEG.trials ~= 1
    error([ ...
        'Resampling and filtering must be performed before epoching. ' ...
        'The current EEG dataset contains %d trials.'], ...
        EEG.trials);
end

% Filtering data containing NaN or Inf can contaminate an entire channel.
% Therefore, confirm that TMS interpolation removed all non-finite values.
if any(~isfinite(EEG.data(:)))
    error([ ...
        'EEG.data still contains NaN or Inf values. ' ...
        'Check the TMS interpolation result before filtering.']);
end

target_sample_frequency = 256;
original_sample_frequency = double(EEG.srate);

fprintf('\n=== Resampling and filtering ===\n');
fprintf('Original sampling frequency: %.3f Hz\n', ...
    original_sample_frequency);

% A raw recording is expected to have a sampling rate greater than 256 Hz.
% Do not silently upsample an already lower-rate processed dataset.
if original_sample_frequency < target_sample_frequency

    error([ ...
        'The current sampling frequency is %.3f Hz, which is lower ' ...
        'than the target frequency of %d Hz. The selected .set file ' ...
        'may already have been downsampled.'], ...
        original_sample_frequency, target_sample_frequency);

elseif abs(original_sample_frequency - ...
        target_sample_frequency) < 1e-9

    fprintf('Data are already sampled at 256 Hz. Resampling skipped.\n');

else

    fprintf('Resampling EEG from %.3f Hz to %d Hz...\n', ...
        original_sample_frequency, target_sample_frequency);

    EEG = pop_resample(EEG, target_sample_frequency);
end

% resampled sample-frequency
sample_frequency = EEG.srate;

EEG = eeg_checkset(EEG);
EEG = eeg_checkset(EEG, 'eventconsistency');

fprintf('Current sampling frequency: %.3f Hz\n', ...
    sample_frequency);

% Confirm that the requested filter frequencies are below Nyquist.
nyquist_frequency = sample_frequency / 2;

notch_low = 59;
notch_high = 61;
bandpass_low = 0.1;
bandpass_high = 100;

if notch_high >= nyquist_frequency
    error([ ...
        'The notch-filter upper cutoff %.1f Hz must be below ' ...
        'the Nyquist frequency %.1f Hz.'], ...
        notch_high, nyquist_frequency);
end

if bandpass_high >= nyquist_frequency
    error([ ...
        'The band-pass upper cutoff %.1f Hz must be below ' ...
        'the Nyquist frequency %.1f Hz.'], ...
        bandpass_high, nyquist_frequency);
end

fprintf('Applying 59-61 Hz notch filter...\n');
EEG = pop_eegfiltnew(EEG, 'locutoff', notch_low, 'hicutoff', notch_high, 'revfilt', 1);
EEG = eeg_checkset(EEG);

fprintf('Applying 0.1-100 Hz band-pass filter...\n');
EEG = pop_eegfiltnew(EEG, 'locutoff', bandpass_low, 'hicutoff', bandpass_high);
EEG = eeg_checkset(EEG);

EEG = eeg_checkset(EEG, 'eventconsistency');

fprintf('Resampling and filtering completed.\n');
fprintf('Final sampling frequency: %.3f Hz\n', EEG.srate);
fprintf('Notch filter: %.1f-%.1f Hz\n', ...
    notch_low, notch_high);
fprintf('Band-pass filter: %.1f-%.1f Hz\n', ...
    bandpass_low, bandpass_high);

%% 3. SAVE FILTERED DATASET

save_name = fullfile(save_folder, [subject '_3_filtered_data.set']);

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', 'Resampled and filtered data', ...
    'savenew', save_name, 'gui', 'off');

fprintf('Saved: %s\n', save_name);

eeglab redraw;


%% 4. TRIALS SEPARATION

% All trials are epoched from the first sound stimulus, stim_1.
%
% Event-label format:
%   task/tms0-or-tms1/target/category/event_role
% Trial subsets are saved according to:
%   subject x task x TMS condition x category

if EEG.trials ~= 1
    error([ ...
        'Trial separation must be performed on continuous EEG data. ' ...
        'The current EEG dataset contains %d epochs.'], ...
        EEG.trials);
end

fprintf('\n=== Trial separation ===\n');
fprintf('Epoch anchor event: stim_1\n');
fprintf('Epoch window: 0 to 1 second after stim_1\n');
fprintf('Current sampling frequency: %.3f Hz\n', EEG.srate);

% pop_resample already updated EEG.event.latency according to EEG.srate.
% Therefore, pop_epoch below automatically uses the resampled event
% latencies and the current resampled sampling frequency.

% Read every structured stim_1 event and determine its:
%   Task
%   TMS condition
%   Category

stim1_event_table = stage4_collect_stim1_events(EEG);

if height(stim1_event_table) == 0
    error([ ...
        'No stim_1 events were found. ' ...
        'Check EEG.event.event_role and the structured events.']);
end
all_stim1_labels = unique(stim1_event_table.EventLabel, 'stable');

fprintf('\n=== Creating all stim_1 epochs ===\n');
fprintf('Number of stim_1 events found: %d\n', height(stim1_event_table));
fprintf('Number of distinct event labels: %d\n', numel(all_stim1_labels));

EEG = pop_epoch(EEG, cellstr(all_stim1_labels), [0 1], ...
    'newname', 'All stim_1 epochs', 'epochinfo', 'yes');

EEG = eeg_checkset(EEG);
EEG = eeg_checkset(EEG, 'eventconsistency');

fprintf('Created epochs: %d\n', EEG.trials);
fprintf('Sampling frequency: %.3f Hz\n', EEG.srate);
fprintf('Samples per epoch: %d\n', EEG.pnts);

filename = string(subject) + "_4_all_epochs.set";
full_save_name = fullfile(save_folder, char(filename));
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname','Set trials','savenew', full_save_name,'gui','off'); 


%% AUTOMATIC BAD-EPOCH DETECTION AND REJECTION

% This section:
%   1. calculates epoch-level QC measurements
%   2. automatically determines subject-specific thresholds
%   3. displays automatically detected bad epochs in EEGPLOT
%   4. automatically rejects them
%   5. saves the QC report and cleaned dataset
%
% No manual epoch selection is required.

if EEG.trials <= 1
    error([ ...
        'Automatic bad-epoch rejection requires epoched EEG data. ' ...
        'The current dataset contains %d trial(s).'], ...
        EEG.trials);
end

fprintf('\n=== Automatic bad-epoch detection ===\n');
fprintf('Epochs before rejection: %d\n', EEG.trials);
fprintf('Channels: %d\n', EEG.nbchan);
fprintf('Samples per epoch: %d\n', EEG.pnts);
fprintf('Sampling frequency: %.3f Hz\n', EEG.srate);

% EEGLAB renumbers epochs after rejection.
% Store an immutable original epoch number for CSV/event alignment.

for epoch_index = 1:EEG.trials
    EEG.epoch(epoch_index).original_epoch = epoch_index;
end

EEG = eeg_checkset(EEG);
EEG = eeg_checkset(EEG, 'eventconsistency');

qc_table = stage5_calculate_bad_epoch_metrics(EEG);

% Automatically calculate subject-specific thresholds
% Robust threshold: median(metric) + robust_z_limit × robust standard deviation
%
% where: robust standard deviation = 1.4826 × MAD
%
% The lower and upper limits prevent the data-derived threshold from
% becoming unrealistically strict or permissive.
%
% These limits assume EEG.data is expressed in microvolts.

robust_z_limit = 6;

[centered_amplitude_threshold, centered_amplitude_median, centered_amplitude_robust_sd] = ...
    stage5_calculate_robust_threshold( ...
        qc_table.MaxCenteredAbs_uV, ...
        robust_z_limit, ...
        NaN, ...
        NaN);

[peak_to_peak_threshold, peak_to_peak_median, peak_to_peak_robust_sd] = ...
    stage5_calculate_robust_threshold( ...
        qc_table.MaxPeakToPeak_uV, ...
        robust_z_limit, ...
        NaN, ...
        NaN);

[step_threshold, step_median, step_robust_sd] = ...
    stage5_calculate_robust_threshold( ...
        qc_table.MaxSampleStep_uV, ...
        robust_z_limit, ...
        NaN, ...
        NaN);

[rms_threshold, rms_median, rms_robust_sd] = ...
    stage5_calculate_robust_threshold( ...
    qc_table.MedianChannelRMS_uV, robust_z_limit, NaN, NaN);

fprintf('\nAutomatically selected thresholds:\n');
fprintf('Maximum centered amplitude: %.3f uV\n', centered_amplitude_threshold);
fprintf('Maximum peak-to-peak:       %.3f uV\n', peak_to_peak_threshold);
fprintf('Median channel RMS:         %.3f uV\n', rms_threshold);
fprintf('Maximum sample step:        %.3f uV/sample\n', step_threshold);

qc_table.FlagCenteredAmplitude = qc_table.MaxCenteredAbs_uV > centered_amplitude_threshold;

qc_table.FlagPeakToPeak = qc_table.MaxPeakToPeak_uV > peak_to_peak_threshold;

qc_table.FlagRMS = qc_table.MedianChannelRMS_uV > rms_threshold;

qc_table.FlagSampleStep = qc_table.MaxSampleStep_uV > step_threshold;


% Define extremely contaminated epochs
% These are rejected even when only one metric exceeds the extreme limit.

% Extremely abnormal epochs are defined relative to the current subject,
% rather than using fixed limits that systematically reject TMS trials.

hard_z_limit = 10;

hard_extreme_flag = ...
    qc_table.MaxCenteredAbs_uV > ...
        centered_amplitude_median + ...
        hard_z_limit * centered_amplitude_robust_sd | ...
    qc_table.MaxPeakToPeak_uV > ...
        peak_to_peak_median + ...
        hard_z_limit * peak_to_peak_robust_sd | ...
    qc_table.MaxSampleStep_uV > ...
        step_median + ...
        hard_z_limit * step_robust_sd;

% Combine the automatic criteria
% Magnitude criteria:
%   large centered amplitude
%   large peak-to-peak range
%
% Shape/noise criteria:
%   elevated RMS
%   abrupt sample-to-sample step
%
% An epoch is rejected when:
%   1. it contains an extreme artifact, or
%   2. both magnitude measurements are abnormal, or
%   3. a magnitude measurement and a shape measurement are abnormal.

magnitude_flag = qc_table.FlagCenteredAmplitude | qc_table.FlagPeakToPeak;

shape_flag = qc_table.FlagRMS | qc_table.FlagSampleStep;

automatic_bad_epoch = hard_extreme_flag | ...
    (qc_table.FlagCenteredAmplitude & qc_table.FlagPeakToPeak) | ...
    (magnitude_flag & shape_flag);

qc_table.HardExtremeFlag = hard_extreme_flag;
qc_table.AutomaticBadEpoch = automatic_bad_epoch;

n_bad_epochs = sum(automatic_bad_epoch);
rejection_percentage = 100 * n_bad_epochs / EEG.trials;

fprintf('\nAutomatic rejection result:\n');
fprintf('Detected bad epochs: %d / %d\n', n_bad_epochs, EEG.trials);
fprintf('Rejection percentage: %.2f%%\n', rejection_percentage);

fprintf('Original epoch numbers marked for rejection:\n');
disp(qc_table.OriginalEpoch(automatic_bad_epoch)');

% Warn about an unusually high rejection rate

if rejection_percentage > 20

    warning([ ...
        'The automatic procedure marked %.2f%% of epochs for rejection. ' ...
        'This is unusually high. Inspect the QC plot and EEGPLOT display ' ...
        'and verify that EEG.data is expressed in microvolts.'], ...
        rejection_percentage);

elseif rejection_percentage == 0

    warning([ ...
        'No bad epochs were detected. Check whether the automatically ' ...
        'selected thresholds are too permissive for this dataset.']);
end

% Save threshold information

threshold_name = [ ...
    "MaxCenteredAbs"; ...
    "MaxPeakToPeak"; ...
    "MedianChannelRMS"; ...
    "MaxSampleStep"];

threshold_value = [ ...
    centered_amplitude_threshold; ...
    peak_to_peak_threshold; ...
    rms_threshold; ...
    step_threshold];

metric_median = [ ...
    centered_amplitude_median; ...
    peak_to_peak_median; ...
    rms_median; ...
    step_median];

metric_robust_sd = [ ...
    centered_amplitude_robust_sd; ...
    peak_to_peak_robust_sd; ...
    rms_robust_sd; ...
    step_robust_sd];

threshold_unit = [ ...
    "uV"; ...
    "uV"; ...
    "uV"; ...
    "uV/sample"];

threshold_table = table( ...
    threshold_name, ...
    threshold_value, ...
    metric_median, ...
    metric_robust_sd, ...
    threshold_unit, ...
    'VariableNames', { ...
        'Metric', ...
        'AutomaticThreshold', ...
        'Median', ...
        'RobustStandardDeviation', ...
        'Unit'});

threshold_file = fullfile(save_folder, [subject '_5_bad_epoch_thresholds.csv']);

writetable(threshold_table, threshold_file);

fprintf('Saved automatic thresholds:\n%s\n', threshold_file);

% Save full epoch QC report

qc_report_file = fullfile(save_folder, [subject '_5_bad_epoch_qc_report.csv']);

writetable(qc_table, qc_report_file);

fprintf('Saved epoch QC report:\n%s\n', qc_report_file);

% Plot metric distributions and automatic thresholds

qc_figure = figure( ...
    'Name', [subject ' automatic bad-epoch thresholds'], ...
    'NumberTitle', 'off', ...
    'Position', [100 100 1200 800]);

subplot(2, 2, 1);

histogram(qc_table.MaxCenteredAbs_uV, 40);
hold on;

current_ylim = ylim;

plot( ...
    [centered_amplitude_threshold ...
     centered_amplitude_threshold], ...
    current_ylim, ...
    '--', ...
    'LineWidth', 1.5);

hold off;

xlabel('Maximum centered amplitude (\muV)');
ylabel('Number of epochs');
title(sprintf( ...
    'Centered amplitude threshold = %.1f \\muV', ...
    centered_amplitude_threshold));
grid on;


subplot(2, 2, 2);

histogram(qc_table.MaxPeakToPeak_uV, 40);
hold on;

current_ylim = ylim;

plot( ...
    [peak_to_peak_threshold ...
     peak_to_peak_threshold], ...
    current_ylim, ...
    '--', ...
    'LineWidth', 1.5);

hold off;

xlabel('Maximum peak-to-peak amplitude (\muV)');
ylabel('Number of epochs');
title(sprintf( ...
    'Peak-to-peak threshold = %.1f \\muV', ...
    peak_to_peak_threshold));
grid on;


subplot(2, 2, 3);

histogram(qc_table.MedianChannelRMS_uV, 40);
hold on;

current_ylim = ylim;

plot( ...
    [rms_threshold rms_threshold], ...
    current_ylim, ...
    '--', ...
    'LineWidth', 1.5);

hold off;

xlabel('Median channel RMS (\muV)');
ylabel('Number of epochs');
title(sprintf( ...
    'RMS threshold = %.1f \\muV', ...
    rms_threshold));
grid on;


subplot(2, 2, 4);

histogram(qc_table.MaxSampleStep_uV, 40);
hold on;

current_ylim = ylim;

plot( ...
    [step_threshold step_threshold], ...
    current_ylim, ...
    '--', ...
    'LineWidth', 1.5);

hold off;

xlabel('Maximum adjacent-sample step (\muV/sample)');
ylabel('Number of epochs');
title(sprintf( ...
    'Sample-step threshold = %.1f \\muV/sample', ...
    step_threshold));
grid on;

qc_figure_file = fullfile( ...
    save_folder, ...
    [subject '_5_bad_epoch_threshold_distributions.fig']);

savefig(qc_figure, qc_figure_file);

qc_png_file = fullfile( ...
    save_folder, ...
    [subject '_5_bad_epoch_threshold_distributions.png']);

saveas(qc_figure, qc_png_file);

fprintf('Saved threshold distribution figures.\n');

% Display automatically detected epochs in EEGPLOT

% Create a preview copy so the displayed data remain unchanged even after
% automatic rejection is applied to EEG.
%
% Automatically detected epochs are superimposed as rejection marks.
% The fourth pop_eegplot argument is zero, so the user is not required to
% manually reject epochs.

EEG_preview = EEG;

EEG_preview.reject.rejmanual = ...
    automatic_bad_epoch';

EEG_preview.reject.rejmanualE = ...
    false(EEG_preview.nbchan, EEG_preview.trials);

EEG_preview.reject.rejmanualE(:, automatic_bad_epoch) = true;

fprintf('\nOpening EEGPLOT preview...\n');
fprintf(['Epochs marked by the automatic procedure are displayed as ' ...
         'rejection candidates.\n']);
fprintf('No manual selection is required.\n');

pop_eegplot(EEG_preview, 1, 1, 0);

drawnow;

% Automatically reject bad epochs

if n_bad_epochs > 0

    EEG = pop_rejepoch( ...
        EEG, ...
        find(automatic_bad_epoch)', ...
        0);

    EEG = eeg_checkset(EEG);
    EEG = eeg_checkset(EEG, 'eventconsistency');
end

fprintf('\nEpochs after automatic rejection: %d\n', ...
    EEG.trials);


save_name = fullfile(save_folder, [subject '_5_bad_epochs_removed.set']);

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', 'Automatic bad epochs removed', ...
    'savenew', save_name, 'gui', 'off');

fprintf('Saved automatically cleaned dataset:\n%s\n', save_name);

eeglab redraw;


%% 6. ICA DECOMPOSITION

if EEG.trials <= 1
    error('ICA decomposition requires epoched EEG data.');
end

if any(~isfinite(EEG.data(:)))
    error('EEG.data contains NaN or Inf values. ICA cannot be performed.');
end

if exist('pop_iclabel', 'file') ~= 2
    error([ ...
        'The ICLabel EEGLAB plugin was not found. ' ...
        'Install or enable ICLabel before running this section.']);
end

fprintf('\n=== ICA decomposition ===\n');
fprintf('Epochs: %d\n', EEG.trials);
fprintf('Channels: %d\n', EEG.nbchan);

EEG = eeg_checkset(EEG);

EEG = pop_runica( ...
    EEG, ...
    'icatype', 'runica', ...
    'extended', 1, ...
    'interrupt', 'on');

EEG = eeg_checkset(EEG);

n_ica_components = size(EEG.icaweights, 1);

fprintf('ICA decomposition completed.\n');
fprintf('ICA components: %d\n', n_ica_components);

EEG = pop_iclabel(EEG, 'default');
EEG = eeg_checkset(EEG);

ic_probabilities = EEG.etc.ic_classification.ICLabel.classifications;

ic_classes = string(EEG.etc.ic_classification.ICLabel.classes);

ic_classes = ic_classes(:);

% The reference pipeline inspected 35 components.
% Automatically evaluate the first 35 components, or all components
% when fewer than 35 are available.

n_components_to_check = min(35, n_ica_components);
components_to_check = (1:n_components_to_check)';

fprintf('Components automatically checked: 1-%d\n', ...
    n_components_to_check);


% Save ICA decomposition before component removal

save_name = fullfile( ...
    save_folder, ...
    [subject '_6_ica_decomposition.set']);

[ALLEEG, EEG, CURRENTSET] = pop_newset( ...
    ALLEEG, EEG, CURRENTSET, ...
    'setname', 'ICA decomposition', ...
    'savenew', save_name, ...
    'gui', 'off');

fprintf('Saved ICA decomposition dataset:\n%s\n', save_name);


%% 7. AUTOMATIC ICA COMPONENT REMOVAL

% Conservatively reject an IC only when ICLabel assigns at least
% 90% probability to one of the artifact classes.
%
% "Other" is not automatically rejected because it may include
% ambiguous neural or experiment-related activity.

artifact_probability_threshold = 0.90;

artifact_class_names = [ ...
    "Muscle"
    "Eye"
    "Heart"
    "Line Noise"
    "Channel Noise"];

artifact_class_columns = zeros( ...
    numel(artifact_class_names), 1);

for class_index = 1:numel(artifact_class_names)

    matching_column = find( ...
        strcmpi(ic_classes, artifact_class_names(class_index)), 1);

    if isempty(matching_column)
        error( ...
            'ICLabel class was not found: %s', ...
            artifact_class_names(class_index));
    end

    artifact_class_columns(class_index) = matching_column;
end

checked_probabilities = ic_probabilities( ...
    components_to_check, ...
    artifact_class_columns);

automatic_reject = any( ...
    checked_probabilities >= artifact_probability_threshold, 2);

bad_components = components_to_check(automatic_reject)';

fprintf('\n=== Automatic ICA component classification ===\n');
fprintf('Artifact probability threshold: %.2f\n', ...
    artifact_probability_threshold);

if isempty(bad_components)

    fprintf('No ICA components met the automatic rejection threshold.\n');

else

    fprintf('Automatically rejected ICA components: %s\n', ...
        mat2str(bad_components));
end

% Save component classification report

[maximum_probability, maximum_class_index] = ...
    max(ic_probabilities, [], 2);

most_likely_class = ...
    ic_classes(maximum_class_index);

component_number = (1:n_ica_components)';

checked_among_first_35 = ...
    component_number <= n_components_to_check;

automatically_removed = ...
    ismember(component_number, bad_components);

classification_report = table( ...
    component_number, ...
    most_likely_class, ...
    maximum_probability, ...
    ic_probabilities(:, 1), ...
    ic_probabilities(:, 2), ...
    ic_probabilities(:, 3), ...
    ic_probabilities(:, 4), ...
    ic_probabilities(:, 5), ...
    ic_probabilities(:, 6), ...
    ic_probabilities(:, 7), ...
    checked_among_first_35, ...
    automatically_removed, ...
    'VariableNames', { ...
        'Component', ...
        'MostLikelyClass', ...
        'MostLikelyProbability', ...
        'BrainProbability', ...
        'MuscleProbability', ...
        'EyeProbability', ...
        'HeartProbability', ...
        'LineNoiseProbability', ...
        'ChannelNoiseProbability', ...
        'OtherProbability', ...
        'CheckedAmongFirst35', ...
        'AutomaticallyRemoved'});

report_file = fullfile( ...
    save_folder, ...
    [subject '_7_ica_component_report.csv']);

writetable(classification_report, report_file);

fprintf('Saved ICA component report:\n%s\n', report_file);


% Automatically remove the rejected components

if ~isempty(bad_components)

    EEG = pop_subcomp( ...
        EEG, ...
        bad_components, ...
        0);

    EEG = eeg_checkset(EEG);
    EEG = eeg_checkset(EEG, 'eventconsistency');
end

EEG.etc.ica_rejection.method = ...
    'ICLabel automatic classification';

EEG.etc.ica_rejection.components_checked = ...
    components_to_check';

EEG.etc.ica_rejection.artifact_probability_threshold = ...
    artifact_probability_threshold;

EEG.etc.ica_rejection.removed_components = ...
    bad_components;

EEG.etc.ica_rejection.n_removed_components = ...
    numel(bad_components);

fprintf('Number of removed ICA components: %d\n', ...
    numel(bad_components));

% Save ICA-cleaned dataset
save_name = fullfile( ...
    save_folder, ...
    [subject '_7_ica_components_removed.set']);

[ALLEEG, EEG, CURRENTSET] = pop_newset( ...
    ALLEEG, EEG, CURRENTSET, ...
    'setname', 'ICA components automatically removed', ...
    'savenew', save_name, ...
    'gui', 'off');

fprintf('Saved ICA-cleaned dataset:\n%s\n', save_name);

eeglab redraw;


%% 8. FINAL DATASET SEPARATION

if EEG.trials <= 1
    error([ ...
        'Final separation requires epoched EEG data. ' ...
        'The current dataset contains %d trial(s).'], ...
        EEG.trials);
end

fprintf('\n=== Final dataset separation ===\n');
fprintf('Subject: %s\n', subject);
fprintf('Available cleaned epochs: %d\n', EEG.trials);

% Preserve the complete ICA-cleaned dataset.
EEG_cleaned = EEG;

% Read the stim_1 label from every remaining epoch.
final_epoch_table = stage8_make_final_epoch_group_table(EEG_cleaned, subject);

fprintf('\n=== Available final trial groups ===\n');

group_count_table = groupsummary( ...
    final_epoch_table, ...
    {'Subject', 'Task', 'TMSCondition', 'Category'});

disp(group_count_table);

group_keys = unique( ...
    final_epoch_table.GroupKey, ...
    'stable');

for group_index = 1:numel(group_keys)

    current_key = group_keys(group_index);

    current_rows = ...
        final_epoch_table.GroupKey == current_key;

    current_group = ...
        final_epoch_table(current_rows, :);

    % Preserve the original cleaned epoch order.
    current_group = sortrows( ...
        current_group, ...
        'Epoch');

    selected_epochs = ...
        current_group.Epoch';

    current_task = ...
        current_group.Task(1);

    current_condition = ...
        current_group.TMSCondition(1);

    current_category = ...
        current_group.Category(1);

    expected_epoch_count = ...
        numel(selected_epochs);

    fprintf('\n--------------------------------------------------\n');
    fprintf('Subject: %s\n', subject);
    fprintf('Task: %s\n', char(current_task));
    fprintf('TMS condition: %s\n', char(current_condition));
    fprintf('Category: %s\n', char(current_category));
    fprintf('Cleaned epochs: %d\n', expected_epoch_count);

    % Select existing cleaned epochs

    EEG_subset = pop_select( ...
        EEG_cleaned, ...
        'trial', ...
        selected_epochs);

    EEG_subset = eeg_checkset(EEG_subset);
    EEG_subset = eeg_checkset( ...
        EEG_subset, ...
        'eventconsistency');

    saved_epoch_count = EEG_subset.trials;

    if saved_epoch_count ~= expected_epoch_count
        error([ ...
            '%s | %s | %s: expected %d epochs, but pop_select ' ...
            'returned %d epochs.'], ...
            char(current_task), ...
            char(current_condition), ...
            char(current_category), ...
            expected_epoch_count, ...
            saved_epoch_count);
    end

    % Dataset name

    dataset_name = ...
        string(subject) + " | " + ...
        current_task + " | " + ...
        current_condition + " | " + ...
        current_category;

    EEG_subset.setname = char(dataset_name);

    % Filename

    task_token = ...
        stage8_make_file_token(current_task);

    condition_token = ...
        stage8_make_file_token(current_condition);

    category_token = ...
        stage8_make_file_token(current_category);

    filename = ...
        string(subject) + ...
        "_8_" + string(task_token) + ...
        "_" + string(condition_token) + ...
        "_" + string(category_token) + ...
        ".set";

    % Save final subset

    EEG_subset = pop_saveset( ...
        EEG_subset, ...
        'filename', char(filename), ...
        'filepath', save_folder);

    fprintf('Saved epochs: %d\n', saved_epoch_count);
    fprintf('Saved: %s\n', ...
        fullfile(save_folder, char(filename)));
end

fprintf('\n=== Final dataset separation completed ===\n');
fprintf('Number of saved groups: %d\n', numel(group_keys));

% Restore the complete ICA-cleaned dataset as the main EEG variable.
EEG = EEG_cleaned;
EEG = eeg_checkset(EEG);

eeglab redraw;
