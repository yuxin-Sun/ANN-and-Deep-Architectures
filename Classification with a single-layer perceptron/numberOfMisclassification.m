function sum_err = numberOfMisclassification(W, X, T)
    % TODO: add pre-check to validate
        % Size of W,X and T
        % Value inside T can ONLY be -1 or 1
   Y = W*X;
   Y(Y>=0)=1; 
   Y(Y<0)=-1;
   error = Y-T; % CORRECT MATCH=0; MISMATCH=-2 OR 2
   sum_err = sum(abs(error))/2;
   % NOTE: this info can only be used for evalution!
end