%% plot_stage3_phonemes_overview.m
% Continuous EEG overview for one phonemes trial
% Show ~12-15 representative channels:
%   - channels with strong pre/post stim_1 difference
%   - while keeping anatomical coverage across scalp regions

clear;
clc;
close all;

%% ============================================================
% 0. EEGLAB
% =============================================================

eeglab_path = getenv('EEGLAB_PATH');
if isempty(eeglab_path)
    error(['Set the EEGLAB_PATH environment variable to your local ' ...
           'EEGLAB installation before running this script.']);
end
addpath(genpath(eeglab_path));
eeglab nogui;

%% ============================================================
% 1. SETTINGS
% =============================================================

subject = 'S02';
data_folder = fullfile('preprocessing_test', subject);

set_file = fullfile(data_folder, sprintf('%s_3_filtered_data.set', subject));

% phonemes task 안에서 몇 번째 stim_1 trial을 볼지
phonemes_trial_index = 1;

pre_time  = 0.5;   % seconds before stim_1
post_time = 1.0;   % seconds after stim_1

% 채널 선택 개수
target_n_channels = 14;

output_folder = fullfile(data_folder, 'figures');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

output_file = fullfile( ...
    output_folder, ...
    sprintf('%s_stage3_phonemes_overview_%02d.png', subject, phonemes_trial_index));

%% ============================================================
% 2. LOAD EEG
% =============================================================

fprintf('\nLoading dataset:\n%s\n\n', set_file);

[file_path, file_name, file_ext] = fileparts(set_file);
EEG = pop_loadset('filename', [file_name file_ext], 'filepath', file_path);
EEG = eeg_checkset(EEG);

if EEG.trials ~= 1
    error('This script requires continuous Stage-3 data. EEG.trials = %d', EEG.trials);
end

fprintf('Subject       : %s\n', subject);
fprintf('Channels      : %d\n', EEG.nbchan);
fprintf('Sampling rate : %.2f Hz\n', EEG.srate);
fprintf('Events        : %d\n', numel(EEG.event));

%% ============================================================
% 3. FIND phonemes/stim_1 EVENTS
% =============================================================

phonemes_stim1_event_indices = [];

for event_index = 1:numel(EEG.event)

    current_event = EEG.event(event_index);

    is_stim1 = false;
    is_phonemes = false;

    % ---------- event_role ----------
    if isfield(current_event, 'event_role')
        role_value = current_event.event_role;
        if iscell(role_value), role_value = role_value{1}; end
        if isstring(role_value), role_value = char(role_value); end
        if ischar(role_value)
            is_stim1 = strcmpi(strtrim(role_value), 'stim_1');
        end
    end

    % ---------- task field ----------
    if isfield(current_event, 'task')
        task_value = current_event.task;
        if iscell(task_value), task_value = task_value{1}; end
        if isstring(task_value), task_value = char(task_value); end
        if ischar(task_value)
            is_phonemes = strcmpi(strtrim(task_value), 'phonemes');
        end
    end

    % ---------- fallback: parse event type ----------
    if isfield(current_event, 'type')
        type_value = current_event.type;
        if iscell(type_value), type_value = type_value{1}; end
        if isnumeric(type_value), type_value = num2str(type_value); end
        if isstring(type_value), type_value = char(type_value); end

        if ischar(type_value)
            type_value = strtrim(type_value);

            if ~is_stim1
                if strcmpi(type_value, 'stim_1') || ~isempty(regexp(type_value, '/stim_1$', 'once'))
                    is_stim1 = true;
                end
            end

            if ~is_phonemes
                if startsWith(lower(type_value), 'phonemes/')
                    is_phonemes = true;
                end
            end
        end
    end

    if is_stim1 && is_phonemes
        phonemes_stim1_event_indices(end+1) = event_index; %#ok<SAGROW>
    end
end

fprintf('\nFound %d phonemes/stim_1 events.\n', numel(phonemes_stim1_event_indices));

if isempty(phonemes_stim1_event_indices)
    error('No phonemes/stim_1 events were found.');
end

if phonemes_trial_index > numel(phonemes_stim1_event_indices)
    error('phonemes_trial_index=%d exceeds number of phonemes/stim_1 events=%d', ...
        phonemes_trial_index, numel(phonemes_stim1_event_indices));
end

%% ============================================================
% 4. SELECT ONE EVENT
% =============================================================

selected_event_index = phonemes_stim1_event_indices(phonemes_trial_index);
selected_event = EEG.event(selected_event_index);

stim1_latency = double(selected_event.latency);
center_sample = round(stim1_latency);

fprintf('Selected phonemes stim_1 event index: %d\n', selected_event_index);
fprintf('Latency: %.3f samples (%.3f s)\n', stim1_latency, (stim1_latency - 1) / EEG.srate);

%% ============================================================
% 5. EXTRACT TIME WINDOW
% =============================================================

pre_samples  = round(pre_time  * EEG.srate);
post_samples = round(post_time * EEG.srate);

start_sample = center_sample - pre_samples;
end_sample   = center_sample + post_samples;

if start_sample < 1
    error('Selected event too close to beginning.');
end
if end_sample > EEG.pnts
    error('Selected event too close to end.');
end

segment = double(EEG.data(:, start_sample:end_sample));

% no additional smoothing / filtering
segment = segment - mean(segment, 2);

time_vector = (-pre_samples:post_samples) ./ EEG.srate;

%% ============================================================
% 6. CHANNEL LABELS
% =============================================================

channel_labels = cell(EEG.nbchan, 1);
for ch = 1:EEG.nbchan
    if isfield(EEG.chanlocs, 'labels') && ~isempty(EEG.chanlocs(ch).labels)
        channel_labels{ch} = EEG.chanlocs(ch).labels;
    else
        channel_labels{ch} = sprintf('Ch%d', ch);
    end
end

%% ============================================================
% 7. SCORE CHANNEL DIFFERENCE (pre vs post)
% =============================================================

% baseline: -300 ~ 0 ms
% response: +50 ~ +250 ms
pre_mask  = (time_vector >= -0.30) & (time_vector < 0);
post_mask = (time_vector >= 0.05)  & (time_vector <= 0.25);

channel_score = zeros(EEG.nbchan, 1);

for ch = 1:EEG.nbchan
    pre_mean  = mean(segment(ch, pre_mask));
    post_mean = mean(segment(ch, post_mask));

    pre_rms   = rms(segment(ch, pre_mask));
    post_rms  = rms(segment(ch, post_mask));

    % mean shift + amplitude change combined
    channel_score(ch) = abs(post_mean - pre_mean) + 0.5 * abs(post_rms - pre_rms);
end

%% ============================================================
% 8. DEFINE SCALP REGIONS
% =============================================================

region_defs = struct();

region_defs.frontal = {'Fp1','Fp2','AF3','AF4','AF7','AF8','F7','F5','F3','F1','Fz','F2','F4','F6','F8'};
region_defs.frontocentral = {'FC5','FC3','FC1','FCz','FC2','FC4','FC6'};
region_defs.central = {'C5','C3','C1','Cz','C2','C4','C6'};
region_defs.temporal_left = {'FT7','T7','TP7'};
region_defs.temporal_right = {'FT8','T8','TP8'};
region_defs.parietal = {'CP5','CP3','CP1','CPz','CP2','CP4','CP6','P7','P5','P3','P1','Pz','P2','P4','P6','P8'};
region_defs.occipital = {'PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};

region_names = fieldnames(region_defs);

%% ============================================================
% 9. PICK CHANNELS:
%    best per region first, then top remaining until target_n
% =============================================================

selected_channels = [];

for r = 1:numel(region_names)
    region_name = region_names{r};
    region_labels = region_defs.(region_name);

    region_idx = find(ismember(lower(channel_labels), lower(region_labels)));

    if ~isempty(region_idx)
        [~, best_local] = max(channel_score(region_idx));
        selected_channels(end+1) = region_idx(best_local); %#ok<SAGROW>
    end
end

selected_channels = unique(selected_channels, 'stable');

% fill up remaining slots using highest-score channels not yet selected
[~, score_order] = sort(channel_score, 'descend');

for k = 1:numel(score_order)
    ch = score_order(k);
    if ~ismember(ch, selected_channels)
        selected_channels(end+1) = ch; %#ok<SAGROW>
    end
    if numel(selected_channels) >= target_n_channels
        break;
    end
end

% sort selected channels roughly by original channel order for clean plotting
selected_channels = sort(selected_channels, 'ascend');

fprintf('\nSelected channels (%d):\n', numel(selected_channels));
disp(strjoin(channel_labels(selected_channels), ', '));

%% ============================================================
% 10. PREPARE PLOT DATA
% =============================================================

plot_data = segment(selected_channels, :);
plot_labels = channel_labels(selected_channels);
n_plot = numel(selected_channels);

abs_values = abs(plot_data(:));
robust_amplitude = prctile(abs_values, 98);
if robust_amplitude == 0 || isnan(robust_amplitude)
    robust_amplitude = 1;
end

display_gain = 2.5;
channel_spacing = robust_amplitude * 2.0;

offsets = ((n_plot - 1):-1:0) * channel_spacing;

%% ============================================================
% 11. OPTIONAL METADATA FOR TITLE
% =============================================================

csv_row_str   = '?';
trial_str     = '?';
stimulus_str  = '?';
phoneme1_str  = '?';
category_str  = '?';

if isfield(selected_event, 'csv_row')
    csv_row_str = num2str(selected_event.csv_row);
elseif isfield(selected_event, 'csvrow')
    csv_row_str = num2str(selected_event.csvrow);
end

if isfield(selected_event, 'trial_n')
    trial_str = num2str(selected_event.trial_n);
elseif isfield(selected_event, 'TrialN')
    trial_str = num2str(selected_event.TrialN);
end

if isfield(selected_event, 'stimulus')
    stimulus_str = local_to_char(selected_event.stimulus);
end

if isfield(selected_event, 'phoneme1')
    phoneme1_str = local_to_char(selected_event.phoneme1);
elseif isfield(selected_event, 'Phoneme1')
    phoneme1_str = local_to_char(selected_event.Phoneme1);
end

if isfield(selected_event, 'category')
    category_str = local_to_char(selected_event.category);
elseif isfield(selected_event, 'Category')
    category_str = local_to_char(selected_event.Category);
end

%% ============================================================
% 12. PLOT
% =============================================================

fig = figure('Color', 'w', 'Position', [80 80 1800 900]);
ax = axes(fig);
hold(ax, 'on');

ylow  = -channel_spacing;
yhigh = n_plot * channel_spacing;

% background
patch(ax, [-pre_time 0 0 -pre_time], [ylow ylow yhigh yhigh], ...
    [0.94 0.96 0.98], 'EdgeColor', 'none');

patch(ax, [0 post_time post_time 0], [ylow ylow yhigh yhigh], ...
    [1.00 0.96 0.90], 'EdgeColor', 'none');

% traces
for i = 1:n_plot
    trace = plot_data(i, :) * display_gain;
    plot(ax, time_vector, trace + offsets(i), ...
        'Color', [0.15 0.28 0.40], 'LineWidth', 1.4);
end

% stim_1 line
xline(ax, 0, 'Color', [0.75 0.10 0.05], 'LineWidth', 2.2);

text(ax, 0, yhigh * 0.97, ' stim\_1', ...
    'Color', [0.75 0.10 0.05], ...
    'FontSize', 14, ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'Rotation', 90);

% axes
yticks(ax, fliplr(offsets));
yticklabels(ax, flipud(plot_labels));

set(ax, ...
    'YDir', 'normal', ...
    'FontSize', 14, ...
    'TickLength', [0 0], ...
    'Box', 'off', ...
    'LineWidth', 1.2);

xlim(ax, [-pre_time post_time]);
ylim(ax, [-channel_spacing * 0.4, offsets(1) + channel_spacing * 0.7]);

xlabel(ax, 'Time from stim\_1 (s)', 'FontSize', 18);
ylabel(ax, 'Selected EEG channels', 'FontSize', 18);

title1 = 'A. Continuous EEG overview';
title2 = sprintf('CSV row: %s | trial %s | stimulus %s | phoneme1 %s | category %s', ...
    csv_row_str, trial_str, stimulus_str, phoneme1_str, category_str);

title(ax, {title1; title2}, 'FontSize', 18, 'FontWeight', 'normal');

% scale bar
scale_uv = 100;
scale_x = post_time - 0.06;
scale_y = channel_spacing * 0.3;

plot(ax, [scale_x scale_x], [scale_y scale_y + scale_uv * display_gain], ...
    'k', 'LineWidth', 2.5);

text(ax, scale_x + 0.015, scale_y + scale_uv * display_gain / 2, ...
    sprintf('%d \\muV', scale_uv), ...
    'FontSize', 14, ...
    'VerticalAlignment', 'middle');

hold(ax, 'off');

exportgraphics(fig, output_file, 'Resolution', 220);

fprintf('\nSaved figure:\n%s\n', output_file);

%% ============================================================
% LOCAL FUNCTION
% =============================================================
function out = local_to_char(x)
    if iscell(x)
        if isempty(x)
            out = '?';
            return;
        end
        x = x{1};
    end
    if isstring(x)
        x = char(x);
    end
    if isnumeric(x)
        out = num2str(x);
    elseif ischar(x)
        if isempty(strtrim(x))
            out = '?';
        else
            out = strtrim(x);
        end
    else
        out = '?';
    end
end
