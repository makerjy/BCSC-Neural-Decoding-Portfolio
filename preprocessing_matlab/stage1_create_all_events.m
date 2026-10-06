
function EEG = stage1_create_all_events(EEG, info_table, info_table_file, summary_table)

    if ~isfield(EEG, 'etc') || isempty(EEG.etc)
        EEG.etc = struct();
    end

    % Archive the source events once. Avoid nesting another archive if this
    % function is accidentally rerun on an already rebuilt dataset.
    if ~isfield(EEG.etc, 'events_before_table_rebuild')
        EEG.etc.events_before_table_rebuild = EEG.event;
        if isfield(EEG, 'urevent')
            EEG.etc.urevents_before_table_rebuild = EEG.urevent;
        end
    end

    EEG.event = struct([]);
    EEG.urevent = struct([]);

    event_roles = [ ...
        "base_line", "task_onset", "tms_1", "tms_2", ...
        "stim_1", "stim_2", "stim_3", "task_final"];

    total_points = EEG.pnts * EEG.trials;

    for row = 1:height(info_table)

        event_latencies = [ ...
            info_table.Base_Line(row), ...
            info_table.Onset_TSidx(row), ...
            info_table.TMS1_TSidx(row), ...
            info_table.TMS2_TSidx(row), ...
            info_table.P1_TSidx(row), ...
            info_table.P2_TSidx(row), ...
            info_table.P3_TSidx(row), ...
            info_table.Final_TSidx(row)];

        for event_index = 1:numel(event_roles)
            latency = event_latencies(event_index);

            % single_phoneme has no P2/P3; phonemes has no P3.
            if ~isfinite(latency)
                continue;
            end

            latency = round(latency);
            if latency < 1 || latency > total_points
                error(['Event outside EEG range at CSV row %d: ' ...
                       '%s latency = %d, valid range = 1..%d.'], ...
                       row, char(event_roles(event_index)), ...
                       latency, total_points);
            end

            EEG = append_one_event( ...
                EEG, info_table, row, event_roles(event_index), latency);
        end
    end

    [~, sort_order] = sort([EEG.event.latency]);
    EEG.event = EEG.event(sort_order);
    EEG = eeg_checkset(EEG, 'eventconsistency');

    EEG.etc.subject = char(info_table.Subject(1));
    EEG.etc.info_table_file = info_table_file;
    EEG.etc.trial_info_table = info_table;
    EEG.etc.trial_summary = summary_table;
    EEG.etc.event_label_schema = ...
        'task/tms0-or-tms1/target/category/event_role';
    EEG.etc.control_tms_note = [ ...
        'For TMS==0 trials, tms_1 and tms_2 are nominal time anchors ' ...
        'relative to P1, not actual TMS pulses.'];
end


function EEG = append_one_event(EEG, info_table, row, event_role, latency)

    task = info_table.TaskCanonical(row);
    tms = info_table.TMSCanonical(row);
    condition = info_table.ConditionCanonical(row);
    target = info_table.TargetCanonical(row);
    % stimulation_site = info_table.StimulationSite(row);
    category = info_table.CategoryCanonical(row);

    event_type = task + "/tms" + string(tms) + "/" + ...
        target + "/" + category + "/" + event_role;

    new_event = struct();
    new_event.latency = latency;
    new_event.duration = 0;
    new_event.type = char(event_type);

    new_event.event_role = char(event_role);
    new_event.task = char(task);
    new_event.tms = tms;
    new_event.condition = char(condition);
    new_event.tms_target = char(target);
    % new_event.stimulation_site = char(stimulation_site);
    new_event.category = char(category);

    new_event.subject = char(info_table.Subject(row));
    new_event.csv_row = info_table.csv_row(row);
    new_event.trial_number = info_table.TrialN(row);
    new_event.stimulus = char(info_table.Stimulus(row));
    new_event.phoneme1 = char(info_table.Phoneme1(row));
    new_event.phoneme2 = char(info_table.Phoneme2(row));
    new_event.phoneme3 = char(info_table.Phoneme3(row));

    if isempty(EEG.event)
        EEG.event = new_event;
    else
        EEG.event(end + 1) = new_event;
    end
end
