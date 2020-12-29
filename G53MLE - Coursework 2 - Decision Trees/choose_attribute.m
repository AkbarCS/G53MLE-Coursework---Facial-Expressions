function [best_feature, best_threshold] = choose_attribute(features, targets)

% set up useful variables
[m, n] = size(features);
total_pn = size(find(targets == 0),1) + size(find(targets == 1),1);
v_array = zeros(1,n - 1);
Gain = zeros(1,n);
Best_value = zeros(1, n);

%loop over attributes
for i = 1 : n
    
    Remainder = 0;
    unique_values = unique(features(:, i));
    for v_index = 1 : (length(unique_values) - 1) % n - 1 times
        
        %choose the value for classification
        v_array(v_index) = (unique_values(v_index) + unique_values(v_index + 1)) / 2;
        
        %calculate the impurity on the left of v_array(v_index)
        indices_left = find(features(:, i) <= v_array(v_index));
        [I_left, sum_pn_left] = calculateI(targets(indices_left));
        remainder_left = sum_pn_left / total_pn * I_left;
        
        %calculate the impurity on the right of v_array(v_index)
        indices_right = find(features(:, i) > v_array(v_index));
        [I_right, sum_pn_right] = calculateI(targets(indices_right));
        remainder_right = sum_pn_right / total_pn * I_right;
        
        Remainder = remainder_left + remainder_right;
        
        gain_in_attr(v_index) = calculateI(targets) - Remainder;  
    end
    
    [Gain(i), best_value_index] = max(gain_in_attr);
    Best_value(i) = v_array(best_value_index);
end
Best_value

%obtain best attribute
[max_Gain, attr_index] = max(Gain);
best_feature = attr_index;

%obtain best threshold
best_threshold = Best_value(attr_index);
end