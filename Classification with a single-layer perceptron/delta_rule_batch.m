function result = delta_rule_batch(X, T, epoch)
% we assume that pattern is 2D,
% otherwiese seperation line/plane cannot visualized
% X = [x1, x2, x3... ; y1, y2, y3...] 
% T = [z1, z2, z3 ...] has the same number of columns as X
    numCols = precheck(X,T,epoch);
    if  numCols < 0    
        result = -1; % Failed
        return; % Skip the left
    end
   
    % Extend X to X_extend
    X_extend=[ X ; ones(1,numCols)];

    % Init W
    W_init=randn(1,3);

    % Learning rate
    eta = 0.1;

    % To be shown, set 0 now
    mismatch= zeros(1,epoch); % In order to show the number Of Misclassification over epochs
        
    W = W_init;    
    %%
    figure
    x_1 = linspace(-5,5,9);

    for index=1:epoch
        delta_W = - eta* (W*X_extend-T)*X_extend';
        W = W + delta_W;
        %%
        mismatch(1,index)=numberOfMisclassification(W, X_extend, T);        
        x_2 = ((-W(1)*x_1)-W(3))/W(2);
        plot(x_1,x_2,'r-') % new line in red color        
        hold on

    end
    %%
    title('Decision Boundaries') 
    xlim([-2,2])
    ylim([-2,2])
    xlabel('x_1 values') % x-axis label
    ylabel('x_2 values') % y-axis label

    figure
    plot(mismatch)
    title('the number Of Misclassification over epochs') 
    xlabel('epochs') % x-axis label
    ylabel('number Of mismatch') % y-axis label
    %%

    result = W; % Proper exit
end

function result = precheck(X,T,epoch)
    if epoch <= 0
        result =-1; % illegual epoch
        return; % Skip the left
    end
    [numRows,numCols]  = size(X);
    if numRows ~= 2
        result = -2;
        return; % Skip the left
    end
    if numCols ~= size(T,2)  %numCols
        result = -3;
        return; % Skip the left
    end
    result = numCols; % Proper exit
end
