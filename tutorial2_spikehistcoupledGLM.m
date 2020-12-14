% tutorial2_spikehistcoupledGLM.m
%
% This is an interactive tutorial designed to walk you through the steps of
% fitting an autoregressive Poisson GLM (i.e., a spiking GLM with
% spike-history) and a multivariate autoregressive Poisson GLM (i.e., a
% GLM with spike-history AND coupling between neurons).
%
% Data: from Uzzell & Chichilnisky 2004; see README file for details. 
%
% Last updated: Mar 10, 2020 (JW Pillow)

% Instructions: Execute each section below separately using cmd-enter.
% For detailed suggestions on how to interact with this tutorial, see
% header material in tutorial1_PoissonGLM.m

%% 1. Load the raw data

% Load data.
data_dir = 'data_RGCs/';      % data directory
load([data_dir, 'Stim']);     % stimulus values (binary white noise)
load([data_dir,'stimtimes']); % stim frame times in seconds (if desired)
load([data_dir, 'SpTimes']);  % load spike times (in units of stim frames)

% Rename variables.
stim = Stim;
stim_ts = stimtimes;
spk_ts = SpTimes;
clear Stim stimtimes SpTimes

% Get basic info on cells and stim.
n_cells = length(spk_ts);           % number of cells
cell_num = 3;                          % cell to load
spk_ts_s_c = spk_ts{cell_num};        % spike timestamps for selected cell
n_spks = length(spk_ts_s_c);          % number of spikes
dt = (stim_ts(2) - stim_ts(1));  % time bin size for stimulus (s)
n_obs = size(stim, 1);                 % number of time bins (observations)

% Print out some basic info.
fprintf('--------------------------\n');
fprintf('Loaded RGC data: cell %d\n', cell_num);
fprintf('Number of stim frames: %d (%.1f minutes)\n', n_obs, ...
        n_obs * (dt / 60));
fprintf('Time bin size: %.5f s\n', dt);
fprintf('Number of spikes: %d (mean rate = %.1f Hz)\n\n', n_spks, ...
        n_spks / n_obs * (1 / dt));

% Plot stimulus values and spike times.
figure
% Plot stimulus values.
subplot(2, 1, 1);
stim_bins_1s = 1 : round(1 / dt);       % bins of stimulus to plot
t_in_stim_bins_1s = stim_bins_1s * dt;  % time bins of stimulus
plot(t_in_stim_bins_1s, stim(stim_bins_1s), 'linewidth', 2); 
axis tight;
title('raw stimulus (full field flicker)');
ylabel('stim intensity');
% Plot spike times.
subplot(2, 1, 2);
% Get the spike times that happen within `bins`.
spk_ts_in_1s = ...
    spk_ts_s_c((spk_ts_s_c >= t_in_stim_bins_1s(1)) ...
                     & (spk_ts_s_c < t_in_stim_bins_1s(end)));
plot(spk_ts_in_1s, 1, 'ko', 'markerfacecolor', 'k');
axis tight
set(gca, 'xlim', t_in_stim_bins_1s([1 end]));
title('spike times');
xlabel('time (s)');

% Let's visualize the spike-train auto and cross-correlations at 0.5s
% around 0.
clf;
n_lags = ceil(0.5 / dt);
for i_cell = 1 : n_cells
    for j_cell = i_cell : n_cells
        % Compute cross-correlation of neuron i with neuron j.
        xc = xcorr(spk_ts{i_cell}, spk_ts{j_cell}, n_lags);

        % remove center-bin correlation value for auto-correlations (for
        % viz purposes)
        if i_cell == j_cell, xc(n_lags + 1) = 0;
        end
        
        % Make plot
        subplot(n_cells, n_cells, ((i_cell-1) * n_cells) + j_cell);
        plot((-n_lags : n_lags) * dt, xc, '.-', 'markersize', 20); 
        axis tight;
        title(sprintf('cells (%d,%d)', i_cell, j_cell));
    end
end
xlabel('time shift (s)');

%% 2. Create the response variables: bin the spike trains

% For now, we will assume we want to use the same bins on the spike train
% as the bins used for the stimulus. Later, though, we'll wish to vary 
% this. This binned spike activity vector (where we have a spike count for
% each observation) will be what we are trying to predict from our GLMs.

% Create time bins for spike train binning.
spk_ts_bins = [0, ((1 : n_obs) * dt)];
% Bin the spike trains.
spk_ts_hist = zeros(n_obs, n_cells);
for i_cell = 1 : n_cells
    spk_ts_hist(:, i_cell) = histcounts(spk_ts{i_cell}, spk_ts_bins);
end
% Rename response variable as `y`.
y = spk_ts_hist;

%% 3. Create the predictor variables: build design matrix

% Set the number of past bins of the stimulus to use for predicting spikes
% (Try varying this, to see how performance changes!)
n_p_x = 25;  % number of parameters in `x` design matrix (bins in past)  
% Set number of time bins of auto-regressive spike history to use.
n_p_a = 24;  % number of parameters for auto-regressive spike history.

% Build stimulus design matrix, `x` (using `hankel`).
% Pad early bins of stimulus with zero.
padded_stim = [zeros(n_p_x-1,1); stim];
x_stim = hankel(padded_stim(1 : (end - n_p_x + 1)), ...
                padded_stim((end - n_p_x + 1) : end));

% Build spike-history design matrix.
padded_spk_ts = [zeros(n_p_a, 1); y(1 : end - 1, cell_num)];
% SUPER important: note that this doesn't include the spike count for the
% bin we're predicting! The spike train is shifted by one bin (back in
% time) relative to the stimulus design matrix.
x_spk_h = hankel(padded_spk_ts(1 : (end - n_p_a + 1)), ...
                 padded_spk_ts((end - n_p_a + 1) : end));

% Combine these into a single design matrix
x = [x_stim, x_spk_h];

% Let's visualize a small part of the design matrix just to see it.
clf
n_obs_disp = 100;
subplot(1, 10, 1:9);
imagesc(1 : (n_p_x + n_p_a), 1 : n_obs_disp, x(1 : n_obs_disp, :));
colormap(gray)
h_cb = colorbar;
h_cb.Location = 'southoutside';
h_cb.Label.String = 'stim intensity value or spike count';
set(gca, 'xticklabel', []);
ylabel('time bin of response');
title('design matrix (including stim and spike history)');
subplot(1, 10, 10); 
imagesc(y(1 : n_obs_disp, cell_num));
h_cb = colorbar;
h_cb.Location = 'southoutside';
set(gca, 'yticklabel', []);
set(gca, 'xticklabel', []);
title('spike count');

% The left part of the design matrix has the stimulus values, the right
% part has the spike-history values. The image on the right is the spike
% count to be predicted (there is a count of one in the 7th bin). Note that
% the spike-history portion of the design matrix is shifted down one 
% relative to the spike count so that we aren't allowed to use the spike
% count on this time bin to predict itself! This can be confirmed below:

obs_first_spk_cell = find(y(:, cell_num), 1, 'first');
assert((x(obs_first_spk_cell, end) == 0) ...
       && x(obs_first_spk_cell + 1, end) == 1);

% Divide the complete dataset into a 60:20:20 training:validation:test 
% ratio, and pick random observations for each subset based on this ratio.
obs_all = [1 : n_obs]';
n_obs_train = ceil(.6 * n_obs);
n_obs_validate = (n_obs - n_obs_train) / 2;
n_obs_test = n_obs_validate;
% Throw error if sum of n_obs in subsets doesn't add up to `n_obs`.
assert((n_obs_train + n_obs_validate + n_obs_test) == n_obs, ...
       ['The number of total observations in the subsets do not match the '...
        'number of total observations in the full dataset.']);
% Set random number generator so we get predictable results.
rng(1);
% Random 60% of data.
obs_train = datasample(obs_all, n_obs_train, 'replace', false);
% Random 20% of the data that has not already been included in `obs_train`.
obs_validate = datasample(setdiff(obs_all, obs_train), n_obs_validate, ...
                          'replace', false);
% Remaining 20% of the data.
obs_test = setdiff(obs_all, [obs_train; obs_validate]);
% Throw error if length of subset of obs doesn't match the n_obs of that
% subset, or if the combined set of unique obs in the subsets doesn't equal
% `obs_all`.
assert(length(obs_train) == n_obs_train ...
       && length(obs_validate) == n_obs_validate ...
       && length(obs_test) == n_obs_test, ...
       'The observations were not properly divided into subsets');
assert(all(sort(unique([obs_train; obs_validate; obs_test])) == obs_all), ...
       'The observations were not properly divided into subsets');
% Set subsets.
x_train = x(obs_train, :);
x_validate = x(obs_validate, :);
x_test = x(obs_test, :);
y_train = y(obs_train, :);
y_validate = y(obs_validate, :);
y_test = y(obs_test, :);

%% 4. Fit & predict with a single-neuron P-GLM with spike history

% <s Fit model on training data, and then predict output for training and
% validation data.

% First fit P-GLM with no spike history.
p_g_l_m = fitglm(x_train(:, 1 : 25), y_train(:, cell_num), ...
                 'distribution', 'poisson', 'link', 'log', ...
                 'intercept', true);
% Get model parameters and predcitions on training and validation sets.             
p_g_l_m_p_train = p_g_l_m.Coefficients.Estimate;
p_g_l_m_y_train = p_g_l_m.Fitted.Response;
p_g_l_m_y_validate = exp([ones(n_obs_validate, 1), x_validate(:, 1 : 25)] ...
                         * p_g_l_m_p_train);

% Then fit P-GLM with spike history ('autoregressive-Poisson GLM')
ap_g_l_m = fitglm(x_train, y_train(:, cell_num), 'distribution', 'poisson',...
                  'link', 'log', 'intercept', true);
ap_g_l_m_p_train = ap_g_l_m.Coefficients.Estimate;
ap_g_l_m_y_train = ap_g_l_m.Fitted.Response;
ap_g_l_m_y_validate = exp([ones(n_obs_validate, 1), x_validate] ...
                          * ap_g_l_m_p_train);
% /s>                      

% <s Visualize parameters (filters)

% Set time bins for plotting
t_bins_stim = ((-n_p_x + 1) : 0) * dt;  % time bins for stim filt
t_bins_spk_h = (-n_p_a : -1) * dt;      % time bins for spike history filt

clf
% Plot stim filters
subplot(2, 1, 1);
hold on
plot(t_bins_stim, p_g_l_m_p_train(2 : end), 'o-');
h = plot(t_bins_stim, ap_g_l_m_p_train(2 : 26), 'o-');
axis tight
legend('P-GLM', 'AP-GLM', 'location', 'northwest');
title('stimulus filters'); 

% Plot spike history filters
subplot(2, 1, 2);
c = h.Color;
h = plot(t_bins_spk_h, ap_g_l_m_p_train(27 : end), 'o-');
axis tight
h.Color = c;
title('spike history filter');
xlabel('time lag (s)');
ylabel('weight');

% Here we see that the stimulus filters for both models share roughly the
% same biphasic shape, with the AP-GLM weights having slightly greater
% magnitude. The spike history filter for the AP-GLM contains only negative
% weights, which tells us that if a spike occurred in the time lag we are
% looking at, one is less likely to occur for the exact observation we are
% looking at. This is particularly true close to the observation (e.g. time
% lags less than -0,02 s), which could be indicative of the cell's
% refractory period.

% /s>

% <s Visualize and quantify fits to training and validation data

% <ss Visualize fits to first second of training and validation data.

clf
% Plot model fits over first second of training data.
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s, 3), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, ap_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
% Plot model fits over first second of validation data.
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s, 3), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, ap_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'P-GLM', 'AP-GLM', ...
       'location', 'northwest');
% /ss>

% <ss Report training performance.

% <sss Compute r-squared values for the models.

% Compute residuals around mean for training and validation sets.
res_train = mean((y_train(:, 3) - mean(y_train(:, 3))) .^ 2);
res_validate = mean((y_validate(:, 3) - mean(y_validate(:, 3))) .^ 2);

% Compute mean-squared error for P-GLM on training set.
p_g_l_m_m_s_e_train = mean((y_train(:, 3) - p_g_l_m_y_train) .^ 2);
% Compute r-squared for P-GLM on training set.
p_g_l_m_r2_train = 1 - (p_g_l_m_m_s_e_train / res_train);
fmt_r2 = ' of the variance in the data is explained by the model.\n';
fprintf(['Training perf (R^2): P-GLM: %.3f', fmt_r2], p_g_l_m_r2_train);
% The r-squared value can also be found directly within the model object.
assert(round(p_g_l_m.Rsquared.Ordinary, 5, 'significant') ...
       == round(p_g_l_m_r2_train, 5, 'significant'));

% Compute mean-squared error for P-GLM on validation set.
p_g_l_m_m_s_e_validate = ...
    mean((y_validate(:, 3) - p_g_l_m_y_validate) .^ 2);
% Compute r-squared for P-GLM on validation set.
p_g_l_m_r2_validate = 1 - (p_g_l_m_m_s_e_validate / res_validate);
fprintf(['Validation perf (R^2): P-GLM: %.3f', fmt_r2], ...
        p_g_l_m_r2_validate);

% Get r-squared for AP-GLM on training set.
ap_g_l_m_r2_train = ap_g_l_m.Rsquared.Ordinary;
fprintf(['Training perf (R^2): AP-GLM: %.3f', fmt_r2], ...
        ap_g_l_m_r2_train);

% Compute mean-squared error for AP-GLM on validation set.
ap_g_l_m_m_s_e_validate = ...
    mean((y_validate(:, 3) - ap_g_l_m_y_validate) .^ 2);
% Compute r-squared for AP-GLM on validation set.
ap_g_l_m_r2_validate = 1 - (ap_g_l_m_m_s_e_validate / res_validate);
fprintf(['Validation perf (R^2): AP-GLM: %.3f', fmt_r2], ...
    ap_g_l_m_r2_validate);
     
% /sss>

% <sss Compute p-values for the models.

% Compute the f-value for P-GLM on training set.
p_g_l_m_fval_train = ((res_train - p_g_l_m_m_s_e_train) / (n_p_x - 1)) ...
                     / (p_g_l_m_m_s_e_train / (n_obs - n_p_x));
% Compute the p-value from the f-distribution for P-GLM on training set.
p_g_l_m_pval_train = ...
    1 - fcdf(p_g_l_m_fval_train, (n_p_x - 1), (n_obs - n_p_x));
fmt_pval = ...
    ['If there is no relationship between our model''s parameters and '...
     'the spike count, then the probability that we would get an R^2 '....
     'value at least as rare as '];
fprintf([fmt_pval, '%.3f for the P-GLM fit on the training data is '...
         '%.5f\n'], p_g_l_m_r2_train, p_g_l_m_pval_train);
% The p-value can also be found directly within the model object.
assert(round(p_g_l_m.devianceTest.pValue(end), 5, 'significant') ...
       == round(p_g_l_m_pval_train, 5, 'significant'));

% Compute the f-value for P-GLM on validation set.
p_g_l_m_fval_validate = ...
    ((res_validate - p_g_l_m_m_s_e_validate) / (n_p_x - 1)) ...
     / (p_g_l_m_m_s_e_validate / (n_obs - n_p_x));
% Compute the p-value from the f-distribution for P-GLM on validation set.
p_g_l_m_pval_validate = ...
    1 - fcdf(p_g_l_m_fval_validate, (n_p_x - 1), (n_obs - n_p_x));
fprintf([fmt_pval, '%.3f for the P-GLM fit on the validation data is '...
         '%.5f\n'], p_g_l_m_r2_validate, p_g_l_m_pval_validate);

% Get the p-value for AP-GLM on training set.
ap_g_l_m_pval_train =  ap_g_l_m.devianceTest.pValue(end);
fprintf([fmt_pval, '%.3f for the AP-GLM fit on the training data is '...
         '%.5f\n'], ap_g_l_m_r2_train, ap_g_l_m_pval_train);

% Compute the f-value for AP-GLM on validation set.
ap_g_l_m_fval_validate = ...
    ((res_validate - ap_g_l_m_m_s_e_validate) / ((n_p_x + n_p_a) - 1)) ...
     / (p_g_l_m_m_s_e_train / (n_obs - (n_p_x + n_p_a)));
% Compute the p-value from the f-distribution for AP-GLM on training set.
ap_g_l_m_pval_validate = ...
    1 - fcdf(ap_g_l_m_fval_validate, ((n_p_x  + n_p_a) - 1), ...
             (n_obs - (n_p_x + n_p_a)));
fprintf([fmt_pval, '%.3f for the AP-GLM fit on the validation data is '...
         '%.5f\n'], ap_g_l_m_r2_validate, ap_g_l_m_pval_validate);     

% Here we see visually and quantitatively that the AP-GLM slightly 
% outperforms the P-GLM.
     
% /sss>
% /ss>
% /s>



%% 5. Fit & predict with a coupled-neuron P-GLM for multi-neuron responses

% <s Build design matrix containing spike history for all neurons

% Create as `n_obs X (n_p_a *  n_cells)`.
x_spk_h_all = zeros(n_obs, (n_p_a * n_cells)); 
for i_cell = 1 : n_cells
    padded_spk_ts = [zeros(n_p_a, 1); spk_ts_hist(1 : (end - 1), i_cell)];
    x_spk_h_all(:, ((i_cell - 1) * n_p_a + 1) : (i_cell * n_p_a)) = ...
        hankel(padded_spk_ts(1 : (end - n_p_a + 1)), ...
               padded_spk_ts((end - n_p_a + 1) : end));
end
% Add stim filter params to get final design matrix.
x_2 = [x_stim, x_spk_h_all];
% Split into training, validation, and test sets.
x_2_train = x_2(obs_train, :);
x_2_validate = x_2(obs_validate, :);
x_2_test = x_2(obs_test, :);
% /s>

% Visualize a small chunk of the design matrix.
clf;
n_obs_disp = 100;
imagesc(1 : 1 : (n_p_x + (n_p_a * n_cells)), 1 : n_obs_disp, ...
        x_2(1 : n_obs_disp , :));
h_cb = colorbar;
h_cb.Location = 'southoutside';
h_cb.Label.String = 'Stim intensity or Spike count';
title('design matrix (stim and 4 neurons spike history)');
xlabel('regressor');
ylabel('time bin of response');
% /s>

% <s Fit the coupled-P-GLM for one neuron.

% Fit coupled-P-GLM on training data and use fit to predict output on
% training and validation data.
cp_g_l_m = ...
    fitglm(x_2_train, y_train(:, cell_num), 'distribution', 'poisson',...
           'link', 'log', 'intercept', true);
cp_g_l_m_p_train = cp_g_l_m.Coefficients.Estimate;
cp_g_l_m_y_train = cp_g_l_m.Fitted.Response;
cp_g_l_m_y_validate = exp([ones(n_obs_validate, 1), x_2_validate] ...
                          * cp_g_l_m_p_train);
% /s>                      

% <s Visualize parameters (filters)

% Set time bins for plotting
t_bins_stim = ((-n_p_x + 1) : 0) * dt;  % time bins for stim filt
t_bins_spk_h = (-n_p_a : -1) * dt;      % time bins for spike history filt

clf
% Plot stim filters
subplot(2, 1, 1);
hold on
plot(t_bins_stim, p_g_l_m_p_train(2 : end), 'o-');
plot(t_bins_stim, ap_g_l_m_p_train(2 : (n_p_x + 1)), 'o-');
plot(t_bins_stim, cp_g_l_m_p_train(2 : (n_p_x + 1)), 'o-');
axis tight
legend('P-GLM', 'AP-GLM', 'CP-GLM', 'location', 'northwest');
title('stimulus filters');

% Plot spike history filters
cell_spk_h_idxs = [(n_p_x + 2) : n_p_a : length(cp_g_l_m_p_train)];
subplot(2, 1, 2);
hold on
for i_cell = 1 : n_cells
    plot(t_bins_spk_h, ...
        cp_g_l_m_p_train(cell_spk_h_idxs(i_cell) ...
        : (cell_spk_h_idxs(i_cell) + n_p_a - 1)), 'o-');
end
axis tight
title('spike history filters');
xlabel('time lag (s)');
ylabel('weight');
legend('cell 1', 'cell 2', 'cell 3', 'cell 4');

% Here we see that the stimulus filter for the AP-GLM is very similar to
% that of the P-GLM. When we look at the spike history filters, we see that
% at time lags close to 0 the predicted output is strongly influenced by
% itself (negatively) and by cell 1 (positively), slightly influenced by 
% cell 2, and barely influenced by cell 4.
% /s>

% <s Visualize and quantify fits to training and validation data.

% <ss Visualize fits to first second of training and validation data.

clf
% Plot model fits over first second of training data.
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s, 3), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, ap_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, cp_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
% Plot model fits over first second of validation data.
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s, 3), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, ap_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, cp_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5); 
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'P-GLM', 'AP-GLM', 'CP-GLM',...
       'location', 'northwest');
% /ss>

% <ss Report training performance.

% <sss Compute r-squared values for the model.

% Compute mean-squared error for CP-GLM on training set.
cp_g_l_m_m_s_e_train = mean((y_train(:, 3) - cp_g_l_m_y_train) .^ 2);
% Compute r-squared for CP-GLM on training set.
cp_g_l_m_r2_train = 1 - (cp_g_l_m_m_s_e_train / res_train);
fprintf(['Training perf (R^2): CP-GLM: %.3f', fmt_r2], cp_g_l_m_r2_train);
% The r-squared value can also be found directly within the model object.
assert(round(cp_g_l_m.Rsquared.Ordinary, 2, 'significant') ...
       == round(cp_g_l_m_r2_train, 2, 'significant'));

% Compute mean-squared error for CP-GLM on validation set.
cp_g_l_m_m_s_e_validate = ...
    mean((y_validate(:, 3) - cp_g_l_m_y_validate) .^ 2);
% Compute r-squared for CP-GLM on validation set.
cp_g_l_m_r2_validate = 1 - (cp_g_l_m_m_s_e_validate / res_validate);
fprintf(['Validation perf (R^2): CP-GLM: %.3f', fmt_r2], ...
        cp_g_l_m_r2_validate);

% /sss>

% <sss Compute p values for the model.

% Compute the f-value for CP-GLM on training set.
cp_g_l_m_fval_train = ...
    ((res_train - cp_g_l_m_m_s_e_train) / (length(cp_g_l_m_p_train) - 1)) ...
     / (cp_g_l_m_m_s_e_train / (n_obs - length(cp_g_l_m_p_train)));
% Compute the p-value from the f-distribution for CP-GLM on training set.
cp_g_l_m_pval_train = ...
    1 - fcdf(cp_g_l_m_fval_train, (length(cp_g_l_m_p_train) - 1), ...
             (n_obs - length(cp_g_l_m_p_train)));
fprintf([fmt_pval, '%.3f for the P-GLM fit on the training data is '...
         '%.5f\n'], cp_g_l_m_r2_train, cp_g_l_m_pval_train);
% The p-value can also be found directly within the model object.
assert(round(cp_g_l_m.devianceTest.pValue(end), 5, 'significant') ...
       == round(cp_g_l_m_pval_train, 5, 'significant'));

% Compute the f-value for CP-GLM on validation set.
cp_g_l_m_fval_validate = ...
    ((res_train - cp_g_l_m_m_s_e_validate) / ...
     (length(cp_g_l_m_p_train) - 1)) ...
     / (cp_g_l_m_m_s_e_validate / (n_obs - length(cp_g_l_m_p_train)));
% Compute the p-value from the f-distribution for CP-GLM on validation set.
cp_g_l_m_pval_validate = ...
    1 - fcdf(cp_g_l_m_fval_validate, (length(cp_g_l_m_p_train) - 1), ...
             (n_obs - length(cp_g_l_m_p_train)));
fprintf([fmt_pval, '%.3f for the CP-GLM fit on the training data is '...
         '%.5f\n'], cp_g_l_m_r2_validate, cp_g_l_m_pval_validate);

% /sss>
% /ss>
% /s>
% 
% Surprisngly, we find that the CP-GLM underperforms relative to the 
% AP-GLM based on the r^2 values. However, we can also compare the fit of 
% the models by comparing their log-likelihood values.

% Compute log-likelihood values of fitted models.
% For P-GLM.
p_g_l_m_ll = y_train(:, 3)' * log(p_g_l_m_y_train) - sum(p_g_l_m_y_train) ...
             - sum(log(factorial(y_train(:, 3))));
% For CP-GLM.         
cp_g_l_m_ll = y_train(:, 3)' * log(cp_g_l_m_y_train) - sum(cp_g_l_m_y_train) ...
              - sum(log(factorial(y_train(:, 3))));          
% These values can also be found directly within the glm object.
assert(ismembertol(p_g_l_m_ll, p_g_l_m.LogLikelihood, 1, 'datascale', 1));
assert(ismembertol(cp_g_l_m_ll, cp_g_l_m.LogLikelihood, 1, ...
                   'datascale', 1));
fprintf('Log-likelihood for P-GLM: %.5f\n', p_g_l_m_ll);
fprintf('Log-likelihood for CP-GLM: %.5f\n', cp_g_l_m_ll);

% Very interestingly, we see that although the CP-GLM had a lower r^2 value
% than the AP-GLM, the CP-GLM has a higher log-likelihood value. It is
% typically rare that an increase in the log-likelihood value is not
% accompanied by an increase in the r^2 value, but it is mathematically
% possible. This raises a philosophical question on which value we should
% use (r^2 or log-likelihood) to select a model when these two values
% disagree (i.e. are not positively correlated). Depending on the context
% and use-case, an argument could be made for either.
%
% At the least, this tells us that, at least for cell 3, adding the
% "coupled spike history parameters" hasn't conclusively improved the
% model.

% So far we've just fit a model with coupling filters for one cell. Let's
% now fit a full population model, where we have a CP-GLM for each of the
% four cells. 

%% 6. Model comparison: log-likelihoood and AIC

% Let's compute loglikelihood (single-spike information) and AIC to see how
% much we gain by adding each of these filter types in turn:

LL_stimGLM = spk_ts(:,cellnum)'*log(ratepred0) - sum(ratepred0);
LL_histGLM = spk_ts(:,cellnum)'*log(ratepred1) - sum(ratepred1);
LL_coupledGLM = spk_ts(:,cellnum)'*log(ratepred2) - sum(ratepred2);

% log-likelihood for homogeneous Poisson model
nsp = sum(spk_ts(:,cellnum));
ratepred_const = nsp/n_obs;  % mean number of spikes / bin
LL0 = nsp*log(ratepred_const) - n_obs*sum(ratepred_const);

% Report single-spike information (bits / sp)
SSinfo_stimGLM = (LL_stimGLM - LL0)/nsp/log(2);
SSinfo_histGLM = (LL_histGLM - LL0)/nsp/log(2);
SSinfo_coupledGLM = (LL_coupledGLM - LL0)/nsp/log(2);

fprintf('\n empirical single-spike information:\n ---------------------- \n');
fprintf('stim-GLM: %.2f bits/sp\n',SSinfo_stimGLM);
fprintf('hist-GLM: %.2f bits/sp\n',SSinfo_histGLM);
fprintf('coupled-GLM: %.2f bits/sp\n',SSinfo_coupledGLM);

% Compute AIC
AIC0 = -2*LL_stimGLM + 2*(1+n_p_x); 
AIC1 = -2*LL_histGLM + 2*(1+n_p_x+n_p_a);
AIC2 = -2*LL_coupledGLM + 2*(1+n_p_x+n_cells*n_p_a);
AICmin = min([AIC0,AIC1,AIC2]); % the minimum of these

fprintf('\n AIC comparison (smaller is better):\n ---------------------- \n');
fprintf('stim-GLM: %.1f\n',AIC0-AICmin);
fprintf('hist-GLM: %.1f\n',AIC1-AICmin);
fprintf('coupled-GLM: %.1f\n',AIC2-AICmin);

% These are whopping differencess! Clearly coupling has a big impact in
% terms of log-likelihood, though the jump from stimulus-only to
% own-spike-history is greater than the jump from spike-history to
% full coupling.


%% Advanced exercises:
% --------------------
% 1. Write code to simulate spike trains from the fitted spike-history GLM.
% Simulate a raster of repeated responses from the stim-only GLM and
% compare to raster from the spike-history GLM

% 2. Write code to simulate the 4-neuron population-coupled GLM. There are
% now 16 spike-coupling filters (including self-coupling), since each
% neuron has 4 incoming coupling filters (its own spike history coupling
% filter plus coupling from three other neurons.  How does a raster of
% responses from this model compare to the two single-neuron models?

% 3. Compute a non-parametric estimate of the spiking nonlinearity for each
% neuron. How close does it look to exponential now that we have added
% spike history? Rerun your simulations using different non-parametric
% nonlinearity for each neuron. How much improvement do you see in terms of
% log-likelihood, AIC, or PSTH % variance accounted for (R^2) when you
% simulate repeated responses?
