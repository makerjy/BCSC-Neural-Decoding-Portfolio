function epoch_table = stage8_make_final_epoch_group_table(EEG, subject)

    n_epochs = EEG.trials;

    Epoch = (1:n_epochs)';
    Subject = repmat(string(subject), n_epochs, 1);

    EventLabel = strings(n_epochs, 1);
    Task = strings(n_epochs, 1);
    TMSCondition = strings(n_epochs, 1);
    Category = strings(n_epochs, 1);

    for epoch_index = 1:n_epochs

        if ~isfield(EEG.epoch, 'eventtype')
            error('EEG.epoch does not contain eventtype.');
        end

        event_types = string( ...
            EEG.epoch(epoch_index).eventtype);

        event_types = event_types(:);
        event_types = strtrim(event_types);

        stim1_index = find( ...
            endsWith(lower(event_types), '/stim_1'));

        if isempty(stim1_index)
            error([ ...
                'Epoch %d does not contain a structured ' ...
                'stim_1 event label.'], ...
                epoch_index);
        end

        if numel(stim1_index) > 1
            error([ ...
                'Epoch %d contains more than one structured ' ...
                'stim_1 event label.'], ...
                epoch_index);
        end

        current_label = ...
            event_types(stim1_index);

        label_parts = split( ...
            current_label, ...
            '/');

        if numel(label_parts) ~= 5
            error([ ...
                'Unexpected event-label format in epoch %d: %s'], ...
                epoch_index, ...
                char(current_label));
        end

        current_task = ...
            lower(strtrim(label_parts(1)));

        current_tms_status = ...
            lower(strtrim(label_parts(2)));

        current_target = ...
            lower(strtrim(label_parts(3)));

        current_category = ...
            lower(strtrim(label_parts(4)));

        % Event format:
        % task/tms0-or-tms1/target/category/stim_1
        %
        % Control trials are grouped as "control".
        % TMS trials are grouped by their target:
        % lip, tongue, ba06, ba44, etc.

        if current_tms_status == "tms0"

            current_condition = "control";

        elseif current_tms_status == "tms1"

            current_condition = current_target;

        else

            error([ ...
                'Unknown TMS status in epoch %d: %s'], ...
                epoch_index, ...
                char(current_tms_status));
        end

        if strlength(current_task) == 0 || ...
                strlength(current_condition) == 0 || ...
                strlength(current_category) == 0

            error( ...
                'Empty grouping value was found in epoch %d.', ...
                epoch_index);
        end

        EventLabel(epoch_index) = current_label;
        Task(epoch_index) = current_task;
        TMSCondition(epoch_index) = current_condition;
        Category(epoch_index) = current_category;
    end

    GroupKey = ...
        Subject + "|" + ...
        Task + "|" + ...
        TMSCondition + "|" + ...
        Category;

    epoch_table = table( ...
        Epoch, ...
        Subject, ...
        Task, ...
        TMSCondition, ...
        Category, ...
        EventLabel, ...
        GroupKey);
end