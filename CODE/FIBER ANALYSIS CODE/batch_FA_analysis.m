%% FA analysis for ALL specified folders (exact MATLAB figs + overlays + full stats)
% Requires: createFit_Single_exp.m, createFit_double_exp.m on the MATLAB path.
% Optional: fitDwelltimehist_poremod2022.m for the Weitz model.

% ROOTS
sciptDir= fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));

% faRootSingle = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\FA SINGLE';
faRootSingle = fullfile(projectRoot,'FA SINGLE');
faRootMulti = fullfile(projectRoot,'FA MULTI');
% faRootMulti  = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\FA MULTI';

% OUTPUT TARGETS (ensure they exist)
faIndGraphs = fullfile(faRootSingle,'FA INDIVIDUAL GRAPHS');   if ~exist(faIndGraphs,'dir'), mkdir(faIndGraphs); end
faIndStats  = fullfile(faRootSingle,'FA INDIVIDUAL STATS');    if ~exist(faIndStats,'dir'),  mkdir(faIndStats);  end
faMulGraphs = fullfile(faRootMulti ,'FA MULTI','FA MULTI GRAPHS'); if ~exist(faMulGraphs,'dir'), mkdir(faMulGraphs); end
faMulStats  = fullfile(faRootMulti ,'FA MULTI','FA MULTI STATS');  if ~exist(faMulStats,'dir'),  mkdir(faMulStats);  end

% Source folders containing .tif images you want to analyze
targetDirs = { ...
  fullfile(projectRoot,'CRYO-SEM DATA','CRYO-SEM X30000','CRYO-SEM X30000 [1]') ...
};

% Make figures render off-screen
oldVis = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off');
cleanupObj = onCleanup(@() set(0,'DefaultFigureVisible',oldVis)); %#ok<NASGU>

for r = 1:numel(targetDirs)
    thisDir = targetDirs{r};
    if ~isfolder(thisDir)
        warning('Source folder not found: %s', thisDir);
        continue;
    end

    datasetLabel = getLastFolderName(thisDir);
    dataTag = sanitizeFigName(datasetLabel);

    tifList = dir(fullfile(thisDir,'*.tif'));
    if isempty(tifList)
        fprintf('[%s] No .tif files.\n', thisDir);
        continue;
    end

    pooled_void_um = [];
    pooled_pxsize  = [];
    pooled_names   = strings(0,1);
    allRows        = table();

    fprintf('\nAnalyzing dataset "%s"\n', datasetLabel);
    fprintf('Source images: %s\n', thisDir);
    fprintf('Saving SINGLE outputs to:\n  %s (graphs)\n  %s (stats)\n', faIndGraphs, faIndStats);
    fprintf('Saving MULTI  outputs to:\n  %s (graphs)\n  %s (stats)\n\n', faMulGraphs, faMulStats);

    for k = 1:numel(tifList)
        fpath = fullfile(thisDir, tifList(k).name);
        baseName  = stripExt(tifList(k).name);
        baseTag   = sanitizeFigName(baseName);
        prefixTag = [dataTag '_' baseTag];

        try
            % --- Load image stack & px size ---
            InfoImage = imfinfo(fpath);
            mImage = InfoImage(1).Width;
            nImage = InfoImage(1).Height;
            N = length(InfoImage);
            Image = zeros(nImage,mImage,N,'uint16');

            if isfield(InfoImage(1),'XResolution') && InfoImage(1).XResolution > 0
                px_size = 1/InfoImage(1).XResolution; % microns per pixel
            else
                error('Missing XResolution in %s', fpath);
            end

            TifLink = Tiff(fpath,'r');
            for ii = 1:N
                TifLink.setDirectory(ii);
                Image(:,:,ii) = TifLink.read();
            end
            TifLink.close();

            FinalImage = Image(:,:,1);
            FinalImage(FinalImage==255) = 1;

            RowToAnalyze = FinalImage'; % analyze across x
            OnOffTh = struct('Ontimes',[],'Offtimes',[]);
            for w = 1:size(RowToAnalyze,2)
                On_diff  = diff([0; RowToAnalyze(:,w); 0] == 1);
                Off_diff = diff([1; RowToAnalyze(:,w); 1] == 0);
                OnbinsStart  = find(On_diff==1);
                OnbinsEnd    = find(On_diff==-1);
                OffbinsStart = find(Off_diff==1);
                OffbinsEnd   = find(Off_diff==-1);
                OnOffTh(w).Ontimes  = OnbinsEnd  - OnbinsStart;
                OnOffTh(w).Offtimes = OffbinsEnd - OffbinsStart;
            end

            All_Voidspace = cell2mat({OnOffTh.Offtimes}');  % pixels
            binsize = 5; % px/bin
            if isempty(All_Voidspace) || max(All_Voidspace) < binsize
                warning('  %s: not enough Off events to build a histogram (skipping).', tifList(k).name);
                continue;
            end

            OffHist = hist(All_Voidspace, binsize:binsize:max(All_Voidspace)); %#ok<HIST>
            OffHist = OffHist(:);
            OffHist_norm = OffHist ./ max(1,numel(All_Voidspace));

            xaxis_px        = (binsize:binsize:max(All_Voidspace))';
            x_axis_units    = xaxis_px .* px_size; % µm
            mean_Off_hist_units = mean(All_Voidspace) .* px_size;

            % --- Capture figs created by fit functions and save to SINGLE/GRAPHS ---
            preFigs = findall(0,'Type','figure');
            [fitresult_single, gof_single] = createFit_Single_exp(x_axis_units, OffHist_norm);
            [fitresult_double, gof_double] = createFit_double_exp(x_axis_units, OffHist_norm);
            newFigs = setdiff(findall(0,'Type','figure'), preFigs);
            for hf = reshape(newFigs,1,[])
                try
                    figName  = get(hf,'Name'); if isempty(figName), figName = 'Figure'; end
                    safeName = sprintf('%s_%s', prefixTag, sanitizeFigName(figName));
                    exportgraphics(hf, fullfile(faIndGraphs, [safeName '.png']), 'Resolution', 300);
                    savefig(hf,      fullfile(faIndGraphs, [safeName '.fig']));
                catch MEfig
                    warning('    (figure save) %s', MEfig.message);
                end
                closeSafely(hf);
            end

            % --- Fit params ---
            p_single = coeffvalues(fitresult_single);
            poresize_single = (-1/p_single(2)); % µm
            p_double = coeffvalues(fitresult_double);
            pore1_double = (-1/p_double(2));    % µm
            pore2_double = (-1/p_double(4));    % µm
            PoreCombined_double = (pore1_double*pore2_double)/(pore1_double+pore2_double);
            fracAmp1 = abs(p_double(1))/(abs(p_double(1))+abs(p_double(3)));
            fracAmp2 = abs(p_double(3))/(abs(p_double(1))+abs(p_double(3)));

            % --- Optional Weitz model ---
            Poresize_Weitz = NaN; cu_curve = [];
            if exist('fitDwelltimehist_poremod2022','file') == 2
                st=1; st_end=numel(OffHist_norm);
                [p_weitz, ~, cu_curve, ~] = fitDwelltimehist_poremod2022( ...
                    x_axis_units(st:st_end), OffHist_norm(st:st_end), 5);
                Poresize_Weitz = p_weitz(2);
            end

            % --- Overlay (SINGLE graph) ---
            fOverlay = figure('Visible','off','Color','w');
            ax = axes('Parent',fOverlay,'YScale','log','YMinorTick','on','FontSize',12);
            hold(ax,'on');
            semilogy(x_axis_units, OffHist_norm, '.', 'DisplayName','Data');
            hS = plot(fitresult_single, x_axis_units, OffHist_norm); set(hS,'DisplayName','Single exponential fit');
            hD = plot(fitresult_double, x_axis_units, OffHist_norm); set(hD,'DisplayName','Double exponential fit');
            if ~isempty(cu_curve)
                semilogy(x_axis_units(1:numel(cu_curve{1})), cu_curve{1}, '-', 'DisplayName','Weitz model fit');
            end
            xlabel('\xi [\mum]','FontSize',14); ylabel('P(\xi)','FontSize',14);
            title(sprintf('%s %s', datasetLabel, baseName), 'Interpreter','none','FontSize',12,'FontWeight','bold');
            legend('Location','northeastoutside'); grid on; hold(ax,'off');
            exportgraphics(fOverlay, fullfile(faIndGraphs, [prefixTag '_HIST_OVERLAY.png']), 'Resolution', 300);
            savefig(fOverlay,        fullfile(faIndGraphs, [prefixTag '_HIST_OVERLAY.fig']));
            closeSafely(fOverlay);

            % --- Residuals (SINGLE graph) ---
            y_fit_double = feval(fitresult_double, x_axis_units);
            residuals = OffHist_norm - y_fit_double;
            fRes = figure('Visible','off','Color','w');
            ax2 = axes('Parent',fRes,'FontSize',12);
            hold(ax2,'on'); plot(x_axis_units, residuals, '-', 'DisplayName','Residuals (Data - Double exp)');
            yline(0,'k:','DisplayName','Zero line');
            xlabel('\xi [\mum]','FontSize',14); ylabel('Residual','FontSize',14);
            title(sprintf('%s %s (Residuals)', datasetLabel, baseName), 'Interpreter','none','FontSize',12,'FontWeight','bold');
            legend('Location','northeastoutside'); grid on; hold(ax2,'off');
            exportgraphics(fRes, fullfile(faIndGraphs, [prefixTag '_RESIDUALS.png']), 'Resolution', 300);
            savefig(fRes,      fullfile(faIndGraphs, [prefixTag '_RESIDUALS.fig']));
            closeSafely(fRes);

            % --- SINGLE stats / per-image exports ---
            stats_struct = struct( ...
                'dataset_label',      string(datasetLabel), ...
                'file_label',         string(baseName), ...
                'filename_full',      string(fpath), ...
                'px_size_um',         px_size, ...
                'mean_Off_hist_um',   mean_Off_hist_units, ...
                'poresize_single_um', poresize_single, ...
                'pore1_double_um',    pore1_double, ...
                'pore2_double_um',    pore2_double, ...
                'PoreCombined_double_um', PoreCombined_double, ...
                'fracAmp1',           fracAmp1, ...
                'fracAmp2',           fracAmp2, ...
                'Poresize_Weitz_um',  Poresize_Weitz, ...
                'gof_single_R2',      safeField(gof_single,'rsquare'), ...
                'gof_double_R2',      safeField(gof_double,'rsquare'), ...
                'n_voids',            numel(All_Voidspace) ...
            );
            T = struct2table(stats_struct);

            writetable(T, fullfile(faIndStats, [prefixTag '_stats.csv']));
            save(fullfile(faIndStats, [prefixTag '_stats.mat']), 'stats_struct');

            H = table(x_axis_units(:), OffHist_norm(:), 'VariableNames',{'xi_um','P_xi'});
            writetable(H, fullfile(faIndStats, [prefixTag '_histogram.csv']));

            pooled_void_um        = [pooled_void_um; (All_Voidspace(:) .* px_size)]; % µm
            pooled_pxsize(end+1,1) = px_size;
            pooled_names(end+1,1)  = string(tifList(k).name);
            allRows = [allRows; T]; %#ok<AGROW>

            fprintf('  %s -> SINGLE outputs saved.\n', tifList(k).name);

        catch ME
            warning('  %s failed: %s', tifList(k).name, ME.message);
        end
    end

    % Combined per-image stats summary (SINGLE)
    if ~isempty(allRows)
        writetable(allRows, fullfile(faIndStats, [dataTag '_ALL_INDIVIDUAL_STATS.csv']));
    end

    % === Dataset-level pooled analysis (MULTI) ===
    if ~isempty(pooled_void_um)
        try
            if numel(unique(round(pooled_pxsize,6))) > 1
                warning('[%s] Mixed pixel sizes; pooling in microns with median-based bins.', datasetLabel);
            end
            binsize_um = median(pooled_pxsize) * 5;  % 5 px (in µm)
            max_um     = max(pooled_void_um);
            edges      = (binsize_um:binsize_um:max_um); if isempty(edges), edges = binsize_um; end
            [counts, centers] = hist(pooled_void_um, edges); %#ok<HIST>
            OffHist_norm_m = counts(:) ./ max(1,numel(pooled_void_um));
            x_axis_units_m = centers(:); % µm

            [fit_single_m, gof_single_m] = createFit_Single_exp(x_axis_units_m, OffHist_norm_m); %#ok<NASGU>
            [fit_double_m, gof_double_m] = createFit_double_exp(x_axis_units_m, OffHist_norm_m); %#ok<NASGU>

            p_single_m = coeffvalues(fit_single_m);
            p_double_m = coeffvalues(fit_double_m);
            pore_single_um   = (-1/p_single_m(2));
            pore1_double_um  = (-1/p_double_m(2));
            pore2_double_um  = (-1/p_double_m(4));
            pore_combined_um = (pore1_double_um*pore2_double_um)/(pore1_double_um+pore2_double_um);
            fracAmp1_m = abs(p_double_m(1))/(abs(p_double_m(1))+abs(p_double_m(3)));
            fracAmp2_m = abs(p_double_m(3))/(abs(p_double_m(1))+abs(p_double_m(3)));

            pore_weitz_um = NaN; cu_multi = [];
            if exist('fitDwelltimehist_poremod2022','file') == 2
                [p_weitz,~,cu_multi,~] = fitDwelltimehist_poremod2022(x_axis_units_m, OffHist_norm_m, 5); %#ok<ASGLU>
                pore_weitz_um = p_weitz(2);
            end

            % MULTI pooled figure
            fpool = figure('Visible','off','Color','w');
            axm = axes('Parent',fpool,'YScale','log','YMinorTick','on','FontSize',12);
            hold(axm,'on');
            semilogy(x_axis_units_m, OffHist_norm_m, '.', 'DisplayName','Data');
            pmS = plot(fit_single_m, x_axis_units_m, OffHist_norm_m); set(pmS,'DisplayName','Single exponential fit');
            pmD = plot(fit_double_m, x_axis_units_m, OffHist_norm_m); set(pmD,'DisplayName','Double exponential fit');
            if ~isempty(cu_multi)
                semilogy(x_axis_units_m(1:numel(cu_multi{1})), cu_multi{1}, '-', 'DisplayName','Weitz model fit');
            end
            xlabel('\xi [\mum]','FontSize',14); ylabel('P(\xi)','FontSize',14);
            title(datasetLabel,'Interpreter','none','FontSize',12,'FontWeight','bold');
            legend('Location','northeastoutside'); grid on; hold(axm,'off');
            exportgraphics(fpool, fullfile(faMulGraphs, [dataTag '_MULTI_POOLED.png']), 'Resolution', 300);
            savefig(fpool,      fullfile(faMulGraphs, [dataTag '_MULTI_POOLED.fig']));
            closeSafely(fpool); closeSafely(fit_single_m); closeSafely(fit_double_m);

            % MULTI pooled stats + histogram
            multiStats = table( ...
                pore_single_um, pore1_double_um, pore2_double_um, pore_combined_um, ...
                fracAmp1_m, fracAmp2_m, pore_weitz_um, ...
                sum(counts)', mean(pooled_void_um), max(pooled_void_um), ...
                'VariableNames', { ...
                    'Pore_single_um','Pore1_double_um','Pore2_double_um','Pore_combined_um', ...
                    'FracAmp1','FracAmp2','Pore_Weitz_um','TotalCounts','MeanVoid_um','MaxVoid_um'} );
            writetable(multiStats, fullfile(faMulStats, [dataTag '_MULTI_STATS.csv']));
            save(fullfile(faMulStats, [dataTag '_MULTI_STATS.mat']), 'multiStats', 'pooled_names');

            Hmulti = table(x_axis_units_m, OffHist_norm_m, 'VariableNames',{'xi_um','P_xi'});
            writetable(Hmulti, fullfile(faMulStats, [dataTag '_MULTI_histogram.csv']));

            fprintf('  pooled summary exported (MULTI).\n');

        catch ME
            warning('[%s] pooled analysis failed: %s', datasetLabel, ME.message);
        end
    end
end

fprintf('\nSINGLE graphs  -> %s\nSINGLE stats   -> %s\n', faIndGraphs, faIndStats);
fprintf('MULTI  graphs  -> %s\nMULTI  stats   -> %s\n', faMulGraphs, faMulStats);

%% helper functions (unchanged)
function nm = stripExt(fn), [~, nm, ~] = fileparts(fn); end
function name = getLastFolderName(p), [~, name] = fileparts(p); end
function v = safeField(s, f), if isstruct(s) && isfield(s, f), v = s.(f); else, v = NaN; end, end
function closeSafely(h)
    if isempty(h), return; end
    if isgraphics(h)
        try, close(h); catch, end
    elseif isvector(h)
        for ii = 1:numel(h)
            if isgraphics(h(ii)), try, close(h(ii)); catch, end, end
        end
    end
end
function s = sanitizeFigName(nm)
    if isempty(nm), nm = 'Figure'; end
    s = regexprep(nm,'[^A-Za-z0-9_\- ]',''); s = strtrim(strrep(s,' ','_'));
end
