function [output] = tree_output(tree, input)
    if isempty(tree.kids)
        output = tree.class;
        return;
    end
    if input(:,tree.op) <= tree.threshold
        output = tree_output(tree.kids{1}, input);
    else
        output = tree_output(tree.kids{2}, input);
    end
end

