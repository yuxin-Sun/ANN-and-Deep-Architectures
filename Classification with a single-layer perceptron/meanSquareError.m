function sum_err = meanSquareError(W, X, T)
    % TODO: add pre-check to validate
        % Size of W,X and T
        % Value inside T should be -1 or 1
   Y = W*X;
   error = Y-T; % Hmmm, the "distance" btw line and point
   sum_err = sum(error.*error)/length(error);
   % NOTE: this info can only be used for evalution!
end