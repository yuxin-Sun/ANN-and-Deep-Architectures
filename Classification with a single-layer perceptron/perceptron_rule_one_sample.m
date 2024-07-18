function W_new = perceptron_rule_one_sample(x, W, t, eta)
    result = precheck(x,W,t);
    if  result < 0   
        W_new = NaN; % Failed
        return; % Skip the left
    end
   y = W*x;
   error = sign(t-y) ;% CORRECT MATCH=0; MISMATCH=-1 OR 1
   W;
   delta_W = eta*error*x';
   W_new = delta_W + W;
end

function result = precheck(x,W,t)

    % X must be one sample. Should be 2+1 Rows
    [numRows,numCols]  = size(x);
    if numRows ~= 3 || numCols ~= 1
        result = -1;
        return; % Skip the left. Wrong x
    end

    [numRows,numCols]  = size(W);
    if numRows ~= 1 || numCols ~= 3
        result = -2;
        return; % Skip the left. Wrong W
    end

    [numRows,numCols]  = size(t);
    if numRows ~= 1 || numCols ~= 1
        result = -3;
        return; % Skip the left. Wrong t
    end
    result = 1; % Proper exit
end