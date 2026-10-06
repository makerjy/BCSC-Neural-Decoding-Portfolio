%% AUDIT WHETHER FIXED AUXILIARY CHANNELS WERE REMOVED
%
% Reference code removes:
%   EOG, M1, M2, BIP1-BIP24
%
% Select S01 or S02 as the known-clean reference.
% Then select the raw S03 dataset to inspect.

clear;
clc;

eeglab nogui;


%% 1. SELECT DATASETS

[reference_file, reference_path] = uigetfile( ...
    '*.set', ...
    'Select a known-clean S01 or S02 .set file');

if isequal(reference_file, 0)
    error('Reference file selection was cancelled.');
end

[target_file, target_path] = uigetfile( ...
    '*.set', ...
    'Select the raw dataset to inspect, for example S03');

if isequal(target_file, 0)
    error('Target file selection was cancelled.');
end


%% 2. LOAD DATASET INFORMATION

% loadmode='info' avoids loading the full EEG signal when supported.
% The fallback loads the full dataset for older EEGLAB versions.

try
    reference_EEG = pop_loadset( ...
        'filename', reference_file, ...
        'filepath', reference_path, ...
        'loadmode', 'info');
catch
    reference_EEG = pop_loadset( ...
        'filename', reference_file, ...
        'filepath', reference_path);
end

try
    target_EEG = pop_loadset( ...
        'filename', target_file, ...
        'filepath', target_path, ...
        'loadmode', 'info');
catch
    target_EEG = pop_loadset( ...
        'filename', target_file, ...
        'filepath', target_path);
end


%% 3. GET CHANNEL LABELS

reference_labels = {reference_EEG.chanlocs.labels};
target_labels = {target_EEG.chanlocs.labels};

reference_canonical = cellfun( ...
    @canonical_channel_label, ...
    reference_labels, ...
    'UniformOutput', false);

target_canonical = cellfun( ...
    @canonical_channel_label, ...
    target_labels, ...
    'UniformOutput', false);


%% 4. CREATE THE EXPECTED REMOVAL LIST

expected_removed_labels = {'EOG', 'M1', 'M2'};

for channel_index = 1:24
    expected_removed_labels{end + 1} = ...
        sprintf('BIP%d', channel_index);
end


%% 5. FIND REMOVAL-TARGET CHANNELS STILL PRESENT IN TARGET

remaining_auxiliary_mask = ismember( ...
    target_canonical, expected_removed_labels);

remaining_auxiliary_labels = ...
    target_labels(remaining_auxiliary_mask);

remaining_auxiliary_canonical = ...
    target_canonical(remaining_auxiliary_mask);


%% 6. COMPARE TARGET AGAINST KNOWN-CLEAN REFERENCE

extra_target_mask = ~ismember( ...
    target_canonical, reference_canonical);

missing_target_mask = ~ismember( ...
    reference_canonical, target_canonical);

extra_in_target = target_labels(extra_target_mask);
missing_from_target = reference_labels(missing_target_mask);


%% 7. CHECK EEGLAB HISTORY

target_history = '';

if isfield(target_EEG, 'history') && ~isempty(target_EEG.history)
    target_history = lower(char(target_EEG.history));
end

history_has_pop_select = contains( ...
    target_history, 'pop_select');

history_has_nochannel = contains( ...
    target_history, 'nochannel');

history_has_rejchan = ...
    contains(target_history, 'pop_rejchan') || ...
    contains(target_history, 'clean_rawdata') || ...
    contains(target_history, 'pop_clean_rawdata') || ...
    contains(target_history, 'clean_channels');

history_has_interpolation = ...
    contains(target_history, 'eeg_interp') || ...
    contains(target_history, 'pop_interp');


%% 8. CHECK EEG.URCHANLOCS

removed_according_to_urchanlocs = {};

if isfield(target_EEG, 'urchanlocs') && ...
        ~isempty(target_EEG.urchanlocs) && ...
        isfield(target_EEG.urchanlocs, 'labels')

    original_labels = {target_EEG.urchanlocs.labels};

    original_canonical = cellfun( ...
        @canonical_channel_label, ...
        original_labels, ...
        'UniformOutput', false);

    removed_mask = ~ismember( ...
        original_canonical, target_canonical);

    removed_according_to_urchanlocs = ...
        original_labels(removed_mask);
end


%% 9. PRINT RESULTS

fprintf('\n');
fprintf('============================================================\n');
fprintf('CHANNEL REMOVAL AUDIT\n');
fprintf('============================================================\n');

fprintf('\nReference dataset\n');
fprintf('  File: %s\n', reference_file);
fprintf('  Set name: %s\n', char(string(reference_EEG.setname)));
fprintf('  Number of channels: %d\n', reference_EEG.nbchan);

fprintf('\nTarget dataset\n');
fprintf('  File: %s\n', target_file);
fprintf('  Set name: %s\n', char(string(target_EEG.setname)));
fprintf('  Number of channels: %d\n', target_EEG.nbchan);

fprintf('\nChannel-count difference\n');
fprintf('  Target minus reference: %+d channels\n', ...
    target_EEG.nbchan - reference_EEG.nbchan);


fprintf('\n------------------------------------------------------------\n');
fprintf('Fixed removal channels still present in target\n');
fprintf('------------------------------------------------------------\n');

fprintf('Count: %d of %d expected removal channels\n', ...
    numel(remaining_auxiliary_labels), ...
    numel(expected_removed_labels));

if isempty(remaining_auxiliary_labels)
    fprintf('  None\n');
else
    for channel_index = 1:numel(remaining_auxiliary_labels)
        fprintf('  %s\n', remaining_auxiliary_labels{channel_index});
    end
end


fprintf('\n------------------------------------------------------------\n');
fprintf('Channels present in target but not reference\n');
fprintf('------------------------------------------------------------\n');

fprintf('Count: %d\n', numel(extra_in_target));

if isempty(extra_in_target)
    fprintf('  None\n');
else
    for channel_index = 1:numel(extra_in_target)
        fprintf('  %s\n', extra_in_target{channel_index});
    end
end


fprintf('\n------------------------------------------------------------\n');
fprintf('Reference channels missing from target\n');
fprintf('------------------------------------------------------------\n');

fprintf('Count: %d\n', numel(missing_from_target));

if isempty(missing_from_target)
    fprintf('  None\n');
else
    for channel_index = 1:numel(missing_from_target)
        fprintf('  %s\n', missing_from_target{channel_index});
    end
end


fprintf('\n------------------------------------------------------------\n');
fprintf('EEGLAB history evidence\n');
fprintf('------------------------------------------------------------\n');

fprintf('  pop_select found: %d\n', history_has_pop_select);
fprintf('  nochannel found: %d\n', history_has_nochannel);
fprintf('  automatic bad-channel rejection found: %d\n', ...
    history_has_rejchan);
fprintf('  channel interpolation found: %d\n', ...
    history_has_interpolation);


fprintf('\n------------------------------------------------------------\n');
fprintf('Channels removed according to EEG.urchanlocs\n');
fprintf('------------------------------------------------------------\n');

if isempty(removed_according_to_urchanlocs)
    fprintf('  No usable removal information found in EEG.urchanlocs.\n');
else
    for channel_index = 1:numel(removed_according_to_urchanlocs)
        fprintf('  %s\n', removed_according_to_urchanlocs{channel_index});
    end
end


%% 10. FINAL INTERPRETATION

fprintf('\n');
fprintf('============================================================\n');
fprintf('INTERPRETATION\n');
fprintf('============================================================\n');

same_channel_set = ...
    isempty(extra_in_target) && ...
    isempty(missing_from_target);

if ~isempty(remaining_auxiliary_labels)

    fprintf([ ...
        'RESULT: The fixed channel-removal step was NOT completed ', ...
        'for the target dataset.\n']);

    fprintf([ ...
        '%d channel(s) from EOG, M1, M2, and BIP1-BIP24 ', ...
        'are still present.\n'], ...
        numel(remaining_auxiliary_labels));

elseif same_channel_set

    fprintf([ ...
        'RESULT: The target has the same channel-label set as ', ...
        'the known-clean reference.\n']);

    fprintf([ ...
        'It is consistent with the fixed auxiliary-channel ', ...
        'removal having been completed.\n']);

elseif isempty(remaining_auxiliary_labels)

    fprintf([ ...
        'RESULT: None of the fixed auxiliary channels remain, ', ...
        'but the channel set differs from the reference.\n']);

    fprintf([ ...
        'The fixed removal may have been completed, but the ', ...
        'dataset uses a different montage or additional EEG ', ...
        'channels were removed or added.\n']);

end

fprintf('\n');


%% LOCAL FUNCTION

function canonical_label = canonical_channel_label(original_label)

    canonical_label = upper(strtrim(char(original_label)));

    % Remove spaces, underscores and hyphens.
    canonical_label = regexprep( ...
        canonical_label, ...
        '[\s_\-]', ...
        '');

    % Normalize BIP01, BIP001, etc. to BIP1.
    bip_token = regexp( ...
        canonical_label, ...
        '^BIP0*([0-9]+)$', ...
        'tokens', ...
        'once');

    if ~isempty(bip_token)
        bip_number = str2double(bip_token{1});
        canonical_label = sprintf('BIP%d', bip_number);
    end

end