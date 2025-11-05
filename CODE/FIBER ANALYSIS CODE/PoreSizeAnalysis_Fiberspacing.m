%% FA SINGLE — batch analysis (no raw image exports, MATLAB indexing fixed)

clear all
close all

% -------- INPUT FOLDER (batch all .tif here) --------
scriptDir= fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));

% folderPath = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\CRYO-SEM DATA\CRYO-SEM X30000\CRYO-SEM X30000 [1]';
folderPath = fullfile(projectRoot,'CRYO-SEM DATA','CRYO-SEM X30000','CRYO-SEM X30000 [1]');
D = dir(fullfile(folderPath, '*.tif'));
filename = {D.name};
N_files  = numel(filename);
folderPath = [folderPath filesep];

if N_files == 0
    error('No .tif files found in: %s', folderPath);
end

% -------- OUTPUT FOLDERS (SINGLE) --------
% outGraphs = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\FA SINGLE\FA INDIVIDUAL GRAPHS';
outGraphs = fullfile(projectRoot,'FA SINGLE','FA INDIVIDUAL GRAPHS');
% outStats  = 'C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\FA SINGLE\FA INDIVIDUAL STATS';
outStats = fullfile(projectRoot,'FA SINGLE','FA INDIVIDUAL STATS');
if ~exist(outGraphs,'dir'), mkdir(outGraphs); end
if ~exist(outStats,'dir'),  mkdir(outStats);  end

% -------- Run invisible figures by default --------
oldVis = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off');
cleanupObj = onCleanup(@() set(0,'DefaultFigureVisible',oldVis)); %#ok<NASGU>

for k = 1:N_files
    try
        fpath   = fullfile(folderPath, filename{k});
        [~, baseName] = fileparts(fpath);
        baseTag = regexprep(strtrim(strrep(baseName,' ','_')), '[^A-Za-z0-9_\-]', '');

        % --- Load stack (first slice only), get px size ---
        InfoImage = imfinfo(fpath);
        mImage = InfoImage(1).Width;
        nImage = InfoImage(1).Height;
        N = numel(InfoImage);
        Image = zeros(nImage, mImage, N, 'uint16');

        if isfield(InfoImage(1),'XResolution') && InfoImage(1).XResolution > 0
            px_size = 1 / InfoImage(1).XResolution;  % µm/px
        else
            error('Missing XResolution in %s', fpath);
        end

        TifLink = Tiff(fpath, 'r');
        for i = 1:N
            TifLink.setDirectory(i);
            Image(:,:,i) = TifLink.read();
        end
        TifLink.close();

        % --- Binarization semantic from your original cell: 255 -> 1 ---
        FinalImage = Image(:,:,1);
        FinalImage(FinalImage == 255) = 1;

        % Analyze across x-direction
        RowToAnalyze = FinalImage';  % columns of original become rows here

        OnOffTh = struct('Ontimes',[],'Offtimes',[]);
        for w = 1:size(RowToAnalyze,2)
            On_diff  = diff([0; RowToAnalyze(:,w); 0] == 1);
            Off_diff = diff([1; RowToAnalyze(:,w); 1] == 0);

            OnbinsStart  = find(On_diff  ==  1);
            OnbinsEnd    = find(On_diff  == -1);
            OffbinsStart = find(Off_diff ==  1);
            OffbinsEnd   = find(Off_diff == -1);

            OnOffTh(w).Ontimes  = OnbinsEnd  - OnbinsStart;
            OnOffTh(w).Offtimes = OffbinsEnd - OffbinsStart;
        end

        % --- Collect void distances (pixels) ---
        All_Voidspace = cell2mat({OnOffTh.Offtimes}');
        if isempty(All_Voidspace)
            warning('No OFF events for %s — skipping.', filename{k});
            continue;
        end

        % --- Histogram (pixels -> µm on x) ---
        binsize_px = 5;
        if max(All_Voidspace) < binsize_px
            warning('Too few OFF counts for %s — skipping.', filename{k});
            continue;
        end
        edges_px = (binsize_px:binsize_px:max(All_Voidspace));
        OffHist  = hist(All_Voidspace, edges_px)'; %#ok<HIST>
        OffHist_norm = OffHist ./ max(1, numel(All_Voidspace));

        xaxis_px     = edges_px(:);
        x_axis_units = xaxis_px .* px_size;      % µm
        mean_Off_hist_units = mean(All_Voidspace) .* px_size;

        % --- Capture figures created by the fit helpers and save them ---
        preFigs = findall(0,'Type','figure');
        [fitresult_single, gof_single] = createFit_Single_exp(x_axis_units, OffHist_norm);
        [fitresult_double, gof_double] = createFit_double_exp(x_axis_units, OffHist_norm);
        newFigs = setdiff(findall(0,'Type','figure'), preFigs);
        for hf = reshape(newFigs,1,[])
            nm = get(hf,'Name'); if isempty(nm), nm = 'Figure'; end
            safe = [baseTag '_' regexprep(strtrim(strrep(nm,' ','_')), '[^A-Za-z0-9_\-]', '')];
            exportgraphics(hf, fullfile(outGraphs, [safe '.png']), 'Resolution', 300);
            savefig(hf,       fullfile(outGraphs, [safe '.fig']));
            close(hf);
        end

        % --- Extract parameters ---
        p_single = coeffvalues(fitresult_single);
        poresize_single = (-1 / p_single(2));  % µm

        p_double = coeffvalues(fitresult_double);
        pore1_double = (-1 / p_double(2));     % µm
        pore2_double = (-1 / p_double(4));     % µm
        PoreCombined_double = (pore1_double * pore2_double) / (pore1_double + pore2_double);
        fracAmp1 = abs(p_double(1)) / (abs(p_double(1)) + abs(p_double(3)));
        fracAmp2 = abs(p_double(3)) / (abs(p_double(1)) + abs(p_double(3)));

        % --- Optional Weitz model (if available) ---
        Poresize_Weitz = NaN; cu_curve = [];
        if exist('fitDwelltimehist_poremod2022','file') == 2
            st = 1; st_end = numel(OffHist_norm);
            [p_weitz, ~, cu_curve, ~] = fitDwelltimehist_poremod2022(x_axis_units(st:st_end), OffHist_norm(st:st_end), 5);
            Poresize_Weitz = p_weitz(2);  % µm
        end

        % --- Overlay: data + fits (log y) ---
        fOverlay = figure('Visible','off','Color','w');
        ax = axes('Parent',fOverlay,'YScale','log','YMinorTick','on','FontSize',12); hold(ax,'on');
        semilogy(x_axis_units, OffHist_norm, '.', 'DisplayName','Data');
        hS = plot(fitresult_single, x_axis_units, OffHist_norm); set(hS,'DisplayName','Single exponential fit');
        hD = plot(fitresult_double, x_axis_units, OffHist_norm); set(hD,'DisplayName','Double exponential fit');
        if ~isempty(cu_curve)
            semilogy(x_axis_units(1:numel(cu_curve{1})), cu_curve{1}, '-', 'DisplayName','Weitz model fit');
        end
        xlabel('\xi [\mum]','FontSize',14); ylabel('P(\xi)','FontSize',14);
        title(baseName,'Interpreter','none','FontSize',12,'FontWeight','bold');
        legend('Location','northeastoutside'); grid on; hold(ax,'off');
        exportgraphics(fOverlay, fullfile(outGraphs, [baseTag '_HIST_OVERLAY.png']), 'Resolution', 300);
        savefig(fOverlay,       fullfile(outGraphs, [baseTag '_HIST_OVERLAY.fig']));
        close(fOverlay);

        % --- Residuals (double exp) ---
        y_fit_double = feval(fitresult_double, x_axis_units);
        residuals = OffHist_norm - y_fit_double;
        fRes = figure('Visible','off','Color','w');
        ax2 = axes('Parent',fRes,'FontSize',12); hold(ax2,'on');
        plot(x_axis_units, residuals, '-', 'DisplayName','Residuals (Data - Double exp)');
        yline(0,'k:','DisplayName','Zero line');
        xlabel('\xi [\mum]','FontSize',14); ylabel('Residual','FontSize',14);
        title([baseName ' (Residuals)'],'Interpreter','none','FontSize',12,'FontWeight','bold');
        legend('Location','northeastoutside'); grid on; hold(ax2,'off');
        exportgraphics(fRes, fullfile(outGraphs, [baseTag '_RESIDUALS.png']), 'Resolution', 300);
        savefig(fRes,       fullfile(outGraphs, [baseTag '_RESIDUALS.fig']));
        close(fRes);

        % --- Save per-image STATS + histogram ---
        T = table( ...
            string(filename{k}), px_size, mean_Off_hist_units, ...
            poresize_single, pore1_double, pore2_double, PoreCombined_double, ...
            fracAmp1, fracAmp2, Poresize_Weitz, ...
            gof_single.rsquare, gof_double.rsquare, numel(All_Voidspace), ...
            'VariableNames', {'file','px_size_um','mean_Off_hist_um', ...
            'poresize_single_um','pore1_double_um','pore2_double_um','PoreCombined_double_um', ...
            'fracAmp1','fracAmp2','Poresize_Weitz_um','gof_single_R2','gof_double_R2','n_voids'} );

        writetable(T, fullfile(outStats, [baseTag '_stats.csv']));
        save(fullfile(outStats, [baseTag '_stats.mat']), 'T');

        H = table(x_axis_units(:), OffHist_norm(:), 'VariableNames',{'xi_um','P_xi'});
        writetable(H, fullfile(outStats, [baseTag '_histogram.csv']));

        fprintf('Done: %s\n', filename{k});

    catch ME
        warning('Failed on %s: %s', filename{k}, ME.message);
    end
end

fprintf('\nSINGLE graphs  -> %s\nSINGLE stats   -> %s\n', outGraphs, outStats);
