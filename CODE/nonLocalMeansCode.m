scriptDir = pwd;
parentDir = fileparts(scriptDir);

I = imread(fullfile(parentDir,'NON-LOCAL MEANS FILTERING','AFM 1% BEFORE NLM.tif'));

filt = imnlmfilt(I, 'DegreeOfSmoothing', 10);

figure; imshowpair(I, filt, 'montage');

imwrite(filt, fullfile(parentDir,'NON-LOCAL MEANS FILTERING','AFTER NLM','nlm_filtered.tif'));