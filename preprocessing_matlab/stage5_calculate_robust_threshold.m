function [threshold, center_value, robust_sd] = ...
        stage5_calculate_robust_threshold( ...
            values, ...
            z_limit, ...
            lower_limit, ...
            upper_limit)

    values = double(values);
    values = values(isfinite(values));

    if isempty(values)
        error('No finite values were provided for threshold calculation.');
    end

    center_value = median(values);

    median_absolute_deviation = ...
        median(abs(values - center_value));

    robust_sd = ...
        1.4826 * median_absolute_deviation;

    if robust_sd <= eps

        threshold = center_value;

    else

        threshold = ...
            center_value + z_limit * robust_sd;
    end

    if ~isnan(lower_limit)
        threshold = max(threshold, lower_limit);
    end

    if ~isnan(upper_limit)
        threshold = min(threshold, upper_limit);
    end
end