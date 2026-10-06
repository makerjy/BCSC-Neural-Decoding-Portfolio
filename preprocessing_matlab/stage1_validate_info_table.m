function stage1_validate_info_table(info_table, subject, EEG)

    task = info_table.TaskCanonical;
    tms = info_table.TMSCanonical;
    target = info_table.TargetCanonical;
    category = info_table.CategoryCanonical;

    allowed_tasks = ["single_phoneme", "phonemes", "words"];
    bad_rows = find(~ismember(task, allowed_tasks));
    assert_no_bad_rows(bad_rows, 'Unknown Task value');

    bad_rows = find(~ismember(tms, [0, 1]) | isnan(tms));
    assert_no_bad_rows(bad_rows, 'TMS must be 0 or 1');

    if startsWith(subject, 'P')
        bad_rows = find(task ~= "phonemes");
        assert_no_bad_rows(bad_rows, ...
            'P-subject tables must contain only phonemes');
    elseif startsWith(subject, 'S')
        expected_tasks = ["single_phoneme", "phonemes", "words"];
        missing_tasks = setdiff(expected_tasks, unique(task));
        if ~isempty(missing_tasks)
            warning('S-subject CSV is missing Task(s): %s', ...
                    strjoin(cellstr(missing_tasks), ', '));
        end
    end

    % single_phoneme: always control; vowels/alveolar/bilabial.
    rows = task == "single_phoneme";

    bad_rows = find(rows & tms ~= 0);
    assert_no_bad_rows(bad_rows, ...
        'single_phoneme must have TMS == 0');

    bad_rows = find(rows & ...
        ~ismember(category, ["vowels", "alveolar", "bilabial"]));
    assert_no_bad_rows(bad_rows, ...
        'single_phoneme has an invalid Category');

    bad_rows = find(rows & ~startsWith(target, "control"));
    assert_no_bad_rows(bad_rows, ...
        'single_phoneme target must be a control label');

    % phonemes: control or TMS on tongue/lip; alveolar/bilabial.
    rows = task == "phonemes";

    bad_rows = find(rows & ...
        ~ismember(category, ["alveolar", "bilabial"]));
    assert_no_bad_rows(bad_rows, ...
        'phonemes has an invalid Category');

    bad_rows = find(rows & tms == 1 & ...
        ~ismember(target, ["tongue", "lip"]));
    assert_no_bad_rows(bad_rows, ...
        'phonemes TMS==1 target must be tongue or lip');

    bad_rows = find(rows & tms == 0 & ...
        ~ismember(target, ["control", "control_tongue", "control_lip"]));
    assert_no_bad_rows(bad_rows, ...
        'phonemes TMS==0 target must be a control label');

    % words: control or TMS on BA06/BA44; real/nonce.
    rows = task == "words";
    
    bad_rows = find(rows & ~ismember(category, ["real", "nonce"]));
    assert_no_bad_rows(bad_rows, ...
        'words has an invalid Category');

    bad_rows = find(rows & tms == 1 & ...
        ~ismember(target, ["ba06", "ba44"]));
    assert_no_bad_rows(bad_rows, ...
        'words TMS==1 target must be BA06 or BA44');

    bad_rows = find(rows & tms == 0 & ...
        ~ismember(target, ["control", "control_ba06", "control_ba44"]));
    assert_no_bad_rows(bad_rows, ...
        'words TMS==0 target must be a control label');

    % P1 and reconstructed analysis windows must fit inside the recording.
    total_points = EEG.pnts * EEG.trials;

    bad_rows = find(~isfinite(info_table.P1_TSidx));
    assert_no_bad_rows(bad_rows, 'P1_TSidx is missing or nonnumeric');

    bad_rows = find(info_table.Base_Line < 1 | ...
                    info_table.Final_TSidx > total_points);
    assert_no_bad_rows(bad_rows, ...
        'A trial window falls outside the selected EEG recording');
end

function assert_no_bad_rows(bad_rows, message_text)
    if ~isempty(bad_rows)
        preview = bad_rows(1:min(20, numel(bad_rows)));
        error('%s. CSV row(s): %s', ...
              message_text, mat2str(preview(:)'));
    end
end