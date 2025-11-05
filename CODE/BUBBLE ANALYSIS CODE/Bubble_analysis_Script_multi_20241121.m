%% Bubble analysis, run multiple images/stacks (BINARY MASKS: pores=black, background=white) — all in one cell

clc
clear all
close all

% Get the script's directory
sciptDir= fileparts(mfilename('fullpath'));

projectRoot = fileparts(fileparts(scriptDir));

% Fixed folder instead of uigetfile (INPUTS)
% filepath = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\CRYO-SEM DATA\CRYO-SEM X30000\CRYO-SEM X30000 [1]';

% Build parts relative to script location
filepath = fullfile(projectRoot,'CRYO-SEM DATA','CRYO-SEM X30000','CRYO-SEM X30000 [1]');

D = dir(fullfile(filepath,'*.tif'));
filename = {D.name};
N_files = numel(filename);
filepath = [filepath filesep];

clear D   % add this line so you can reuse D later as a cell array

nspacing = 1;

% === OUTPUT SUBFOLDERS (absolute GitHub paths) ===
% githubOutRoot_multi = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\BA [MULTI]';
githubOutRoot_multi = fullfile(projectRoot,'BA [MULTI]');
out_multi_graphs = fullfile(githubOutRoot_multi, 'BA_MULTI_GRAPHs');
out_multi_stats  = fullfile(githubOutRoot_multi, 'BA_MULTI_STATs');
if ~exist(out_multi_graphs,'dir'), mkdir(out_multi_graphs); end
if ~exist(out_multi_stats,'dir'),  mkdir(out_multi_stats);  end

g=1;
for i=1:N_files
    fname = [filepath filename{i}];

    [All_bubbles_stack, px_size, Porosity, Pore_coverage] = ...
        Bubble_analysis_from_binary_masks(fname, nspacing);

    FinalBubble_radii{g}=All_bubbles_stack;
    Porosity_all{g}=Porosity;
    Pore_Coverage_all{g}=Pore_coverage;

    FinalBubble_diameters{g}=All_bubbles_stack.*px_size.*2;
    px_size_all{g}=px_size;

    TotalNofBubbles(g)=numel(FinalBubble_diameters{g});

    Avg_Poresize(g)=mean(FinalBubble_diameters{g});
    Std_Poresize(g)=std(FinalBubble_diameters{g});
    std_avg_ratio(g)=Std_Poresize(g)/Avg_Poresize(g);
    Median_Poresize(g)=median(FinalBubble_diameters{g});

    Avg_Porosity(g)=mean(Porosity_all{g});
    Std_Porosity(g)=std(Porosity_all{g});

    Avg_Pore_coverage(g)=mean(Pore_Coverage_all{g});
    Std_Pore_coverage(g)=std(Pore_Coverage_all{g});

    % Per-image stats & diameters 
    [~, base, ~] = fileparts(filename{g});
    T_indiv = table( ...
        Avg_Poresize(g), Std_Poresize(g), Median_Poresize(g), std_avg_ratio(g), ...
        Avg_Porosity(g), Avg_Pore_coverage(g), TotalNofBubbles(g), ...
        'VariableNames', {'Mean','SD','Median','SD_Mean_ratio','Porosity','Pore_Coverage','Total_N_bubbles'} );
    writetable(T_indiv, fullfile(out_multi_stats, [base '_stats.xlsx']));
    writetable(T_indiv, fullfile(out_multi_stats, [base '_stats.csv']));
    writematrix(FinalBubble_diameters{g}(:), fullfile(out_multi_stats, [base '_diameters_um.csv']));
    
    g=g+1;
end

% print out
nspacing
filename

Avg_Poresize
Std_Poresize
std_avg_ratio
Median_Poresize

% Plot individual normalized histograms (and save each)
g=1;
for i=1:N_files
    figure
    hold on
    box on

    b{g}=hist(FinalBubble_radii{i},[1:1:ceil(max(FinalBubble_radii{i}))]);
    P{g}=b{g}./sum(b{g});
    xx{g}=0:numel(b{g})-1;
    D{g}=xx{g}.*2.*px_size_all{g};
    plot(D{g},P{g},'.k--','MarkerSize',10)
    title({filename{g}},'FontSize',10)
    xlabel({'D (\mum)'},'FontSize',14);
    ylabel({'P(D)'},'FontSize',14);
    grid on

    % Save per-image histogram figure
    [~, base, ~] = fileparts(filename{g});
    exportgraphics(gcf, fullfile(out_multi_graphs, [base '_indiv_hist_view.png']), 'Resolution', 300);
    saveas(gcf, fullfile(out_multi_graphs, [base '_indiv_hist_view.fig']));
  
    hold off
    g=g+1;
end

% Plot all results in normalized histograms (legend outside, 11 distinct colors)
figure
set(gcf,'Position',[100 100 950 500]); % wider to fit the outside legend
hold on
box on

% Define 11 distinct colors (visually distinct)
cmap = [ ...
    0.000 0.447 0.741;  % blue
    0.850 0.325 0.098;  % orange-red
    0.929 0.694 0.125;  % yellow
    0.494 0.184 0.556;  % purple
    0.466 0.674 0.188;  % green
    0.301 0.745 0.933;  % light blue
    0.635 0.078 0.184;  % dark red
    0.000 0.500 0.000;  % dark green
    0.750 0.500 0.000;  % brownish orange
    0.250 0.250 0.250;  % dark gray
    0.900 0.600 0.900]; % light magenta

g = 1;
for i = 1:N_files
    color_i = cmap(mod(i-1, size(cmap,1)) + 1, :);

    b{g}  = hist(FinalBubble_radii{i}, [1:1:ceil(max(FinalBubble_radii{i}))]);
    P{g}  = b{g} ./ sum(b{g});
    xx{g} = 0:numel(b{g})-1;
    D{g}  = xx{g} .* 2 .* px_size_all{g};

    plot(D{g}, P{g}, 'MarkerSize', 10, 'Marker', '.', 'LineStyle', '--', 'Color', color_i);
    leg(g) = {filename{g}}; %#ok<AGROW>
    g = g + 1;
end

lgd = legend(leg, 'Location', 'eastoutside', 'Interpreter', 'none');
xlabel({'D (\mum)'}, 'FontSize', 14);
ylabel({'P(D)'}, 'FontSize', 14);
grid on
hold off

% Save summary line plot 
exportgraphics(gcf, fullfile(out_multi_graphs, 'ALL_normalized_histograms.png'), 'Resolution', 300);
saveas(gcf, fullfile(out_multi_graphs, 'ALL_normalized_histograms.fig'))

set(gca, 'LooseInset', get(gca, 'TightInset'));

figure
hold on
box on
if exist('boxplotGroup','file') == 2
    boxplotGroup(FinalBubble_diameters, 'PrimaryLabels', filename);
else
    allD = []; grp = [];
    for ii = 1:N_files
        di = FinalBubble_diameters{ii}(:);
        allD = [allD; di];
        grp = [grp; repmat(ii, numel(di), 1)];
    end
    boxplot(allD, grp, 'Labels', filename, 'LabelOrientation', 'inline');
end
ylabel({'D (\mum)'},'FontSize',14);
set(gca,'TickLabelInterpreter','none'); 
xtickangle(45);
hold off

% Save grouped boxplot
exportgraphics(gcf, fullfile(out_multi_graphs, 'ALL_boxplots.png'), 'Resolution', 300);
saveas(gcf, fullfile(out_multi_graphs, 'ALL_boxplots.fig'));

%%%%%% Results array (Rows: avg, std, median, sd/avg, Porosity, Pore_Coverage, Total N)
ResultsArray=[Avg_Poresize; Std_Poresize; Median_Poresize; ...
              std_avg_ratio; Avg_Porosity; Avg_Pore_coverage; TotalNofBubbles]';
ResultsTable=array2table(ResultsArray);
ResultsTable.Properties.VariableNames = ...
    {'Mean','SD','Median','SD_Mean_ratio','Porosity','Pore_Coverage','Total_N_bubbles'};
ResultsTable.Properties.RowNames = filename;

% Save to BA [MULTI]\BA_MULTI_STATs
writetable(ResultsTable, fullfile(out_multi_stats, 'ResultsTable.xlsx'), 'WriteRowNames', true);
writetable(ResultsTable, fullfile(out_multi_stats, 'ResultsTable.csv'),  'WriteRowNames', true);

% Helper functions ... (unchanged)
%   Bubble_analysis_from_binary_masks, pixel_size_from_tags, fov_from_metadata_or_name
