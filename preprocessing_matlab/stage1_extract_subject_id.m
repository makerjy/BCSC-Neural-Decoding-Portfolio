
function subject = stage1_extract_subject_id(file_base_name)
    subject = regexp(file_base_name, '^[PSps]\d+', 'match', 'once');
    if isempty(subject)
        error('Could not extract a P## or S## subject ID from: %s', ...
              file_base_name);
    end
    subject = upper(subject);
end