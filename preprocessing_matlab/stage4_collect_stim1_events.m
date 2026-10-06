function stim1_table = stage4_collect_stim1_events(EEG)
% COLLECT_STIM1_EVENTS
%
% Reads structured stim_1 events from continuous EEG.
%
% Expected event-label structure:
%
%   task/tms0-or-tms1/target/category/stim_1
%
% TMS condition is determined as follows:
%
%   tms0 -> control
%   tms1 -> use the target field
%
% Examples:
%
%   phonemes/tms0/lip/alveolar/stim_1
%       -> Condition = control
%
%   phonemes/tms1/lip/alveolar/stim_1
%       -> Condition = lip
%
%   words/tms1/ba44/real/stim_1
%       -> Condition = ba44

    n_events = numel(EEG.event);

    event_labels = strings(0, 1);
    tasks = strings(0, 1);
    tms_targets = strings(0, 1);
    categories = strings(0, 1);

    resampled_latencies = zeros(0, 1);
    source_event_indices = zeros(0, 1);

    for event_index = 1:n_events

        event_label = EEG.event(event_index).type;
        task_value =  EEG.event(event_index).task;
        tms_condition_value =  EEG.event(event_index).tms_target;
        category_value =  EEG.event(event_index).category;

        event_role = EEG.event(event_index).event_role;
        
        % stim_1 is always the epoch anchor.
        if event_role ~= "stim_1"
            continue;
        end


        if task_value == ""
            error('A stim_1 event has an empty task value: %s',char(event_label));
        end

        if category_value == ""
            error('A stim_1 event has an empty category value: %s', char(event_label));
        end

        % Normalize the TMS condition.
        if tms_condition_value == ""
            error('A tms1 stim_1 event has an empty target value: %s', char(event_label));
        end

        current_latency = double(EEG.event(event_index).latency);
        if ~isfinite(current_latency)
            error('A stim_1 event has an invalid latency: %s', char(event_label));
        end

        event_labels(end + 1, 1) = event_label;
        tasks(end + 1, 1) = task_value;
        tms_targets(end + 1, 1) = tms_condition_value;
        categories(end + 1, 1) = category_value;

        resampled_latencies(end + 1, 1) = current_latency;

        source_event_indices(end + 1, 1) = event_index;
    end

    group_keys = ...
        tasks + "|" + ...
        tms_targets + "|" + ...
        categories;

    stim1_table = table( ...
        event_labels, ...
        tasks, ...
        tms_targets, ...
        categories, ...
        resampled_latencies, ...
        source_event_indices, ...
        group_keys, ...
        'VariableNames', { ...
            'EventLabel', ...
            'Task', ...
            'TMSCondition', ...
            'Category', ...
            'ResampledLatency', ...
            'SourceEventIndex', ...
            'GroupKey'});
end

