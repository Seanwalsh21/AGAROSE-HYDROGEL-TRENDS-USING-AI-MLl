scriptDir = pwd;
parentDir = fileparts(scriptDir);


T = readtable(fullfile(parentDir,'CRYO-SEM DATA','CRYO-SEM X30000','CRYO-SEM X30000 [1]','CRYO-SEM X30000 [1] STATS','GOLD STANDARD [X30000] Results.csv'));

x = T.X; y = T.Y; n = size(T,1);

% calc euclidean distance from origin for each pore
euc_dist = sqrt(x.^2 + y.^2);

% save individual distances
dist_table = table((1:n)', x, y, euc_dist, 'VariableNames', {'PoreID','X','Y','EuclideanDistance'});
writetable(dist_table, fullfile(parentDir,'MATLAB STATS','EUCLEDIAN DISTANCE','euclidean_dist.csv'));

% stats on everything numeric except coords
cols = T.Properties.VariableNames;
ignore = {'x','X','Y','FeretX','FeretY','FeretAngle','Angle'};
stats = {};

for k=1:length(cols)
    if isnumeric(T.(cols{k})) && ~ismember(cols{k}, ignore)
        vals = T.(cols{k});
        m = mean(vals,'omitnan');
        s = std(vals,'omitnan');
        df = sum(~isnan(vals))-1;
        err = tinv(0.975,df) * s/sqrt(df+1);
        stats{end+1,1} = cols{k};
        stats{end,2} = m;
        stats{end,3} = m-err;
        stats{end,4} = m+err;
    end
end

% add area equivalent diameter
aecd = 2*sqrt(T.Area/pi);
m = mean(aecd); s = std(aecd); df = length(aecd)-1;
err = tinv(0.975,df) * s/sqrt(df+1);
stats{end+1,1} = 'AECD';
stats{end,2} = m; stats{end,3} = m-err; stats{end,4} = m+err;

% add euclidean distance stats
m = mean(euc_dist); s = std(euc_dist); df = length(euc_dist)-1;
err = tinv(0.975,df) * s/sqrt(df+1);
stats{end+1,1} = 'EuclideanDistance';
stats{end,2} = m; stats{end,3} = m-err; stats{end,4} = m+err;

% write summary
summary = cell2table(stats, 'VariableNames', {'var','mean','lo','hi'});
writetable(summary, fullfile(parentDir,'MATLAB STATS','STATS','stats.csv'));

fprintf('%d pores analyzed\n', n);