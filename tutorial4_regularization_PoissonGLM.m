% tutorial4_regularization_PoissonGLM.m
%
% This is an interactive tutorial covering regularization for Poisson GLMs,
% namely maximum a priori ('MAP') estimation of the linear filter
% parameters under a Gaussian prior.
%
% We'll consider two simple regularization methods:
%
% 1. Ridge regression - corresponds to maximum a posteriori (MAP)
%                       estimation under an iid Gaussian prior on the
%                       filter coefficients. 
%
% 2. L2 smoothing prior - corresponds to an iid Gaussian prior on the
%                         pairwise-differences of the filter(s).
%
% Data: from Uzzell & Chichilnisky 2004; see README file for details. 
%
% Last updated: Mar 10, 2020 (JW Pillow)

% Tutorial instructions: Execute each section below separately using
% cmd-enter. For detailed suggestions on how to interact with this
% tutorial, see header material in tutorial1_PoissonGLM.m

%% 1. Load the raw data

addpath GLMtools; % add directory with log-likelihood functions

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
n_cells = length(spk_ts);        % number of cells
cell_num = 3;                    % selected cell to load
spk_ts_s_c = spk_ts{cell_num};   % spike timestamps for selected cell
n_spks = length(spk_ts_s_c);     % number of spikes
dt = (stim_ts(2) - stim_ts(1));  % time bin size for stimulus (s)
n_obs = size(stim, 1);           % number of time bins (observations)

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
% Pick a cell to work with

%% 2. Create the (upsampled) response variable

% The need to regularize GLM parameter estimates is acute when using
% correlated (e.g. naturalistic) stimuli - since the stimuli don't have
% enough power at all frequencies to estimate all frequency components of
% the filter - or when we don't have enough data relative to the number of
% parameters we're trying to estimate. To simulate such a setting we will
% consider the binary white-noise stimulus sampled on a finer time lattice
% than the original stimulus.

% For speed of our code and to illustrate the advantages of regularization,
% let's use only a reduced subset of the dataset.
t_sub = 2;  % subset of time to use (in minutes)
n_obs_sub = round((1 / dt) * 60 * t_sub);
stim_sub = stim(1 : n_obs_sub);
spk_ts_sub = cellfun(@(z) z(z < (t_sub * 60)), spk_ts, 'uni', 0);

% Upsample to finer time scale.
upsamp_x = 5;  % upsample factor
dt_u = dt / upsamp_x;  % dt upsampled
t_sub_u = (dt_u : dt_u : round(n_obs_sub * dt));  % time subset, upsampled
stim_sub_u = interp1((1 : n_obs_sub) * dt, stim_sub, t_sub_u, ...
                     'nearest', 'extrap');  % stim subset, upsampled
stim_sub_u = stim_sub_u(:);  % ensure col vector                 
n_obs_sub_u = length(t_sub_u);  % n_obs subset, upsampled

% Visualize 1 s of the upsampled data.
clf;
% Plot upsampled stimulus.
subplot(2, 1, 1);
stim_bins_1s_u = 1 : ceil(1 / dt_u);         % bins of stimulus to plot
t_in_stim_bins_1s_u = stim_bins_1s_u * dt_u;  % time bins of stimulus
plot(t_in_stim_bins_1s_u, stim_sub_u(stim_bins_1s_u), 'linewidth', 2);
axis tight;
title('raw stimulus - fine time bins (full field flicker)');
ylabel('stim intensity');
% Bin the spike trains and plot binned counts for selected cell.
spk_ts_hist = zeros(n_obs_sub_u, n_cells);
spk_ts_bins = [0, ((1 : n_obs_sub_u) * dt_u)];
for i_cell = 1 : n_cells
    spk_ts_hist(:, i_cell) = histcounts(spk_ts{i_cell}, spk_ts_bins);
end
subplot(2, 1, 2);
stem(t_in_stim_bins_1s_u, spk_ts_hist(stim_bins_1s_u, cell_num));
title('binned spike counts');
ylabel('spike count'); 
xlabel('time (s)');
axis tight;
% Rename response variable as `y`.
y = spk_ts_hist;

% We see now that we have a maximum of 1 spike per bin - we can represent
% this as a binomial process.

% Ensure binary response.
y(y > 1) = 1;

%%  3. Divide data into "training" and "test" sets for cross-validation

% ----------------------------------------------------------------------- %
% We'll build the design matrix to contain a stimulus filter and filters
% of the outputs (spike history) for each of the cells.

n_p_s = 125;  % number of parameters (bins in past) for stimulus filter  
n_p_a = 120;  % number of parameters (bins in past) for spike history

% Create spike history portion of design matrix as:
% `n_obs X (n_p_a *  n_cells)`.
x_spk_h_all = zeros(n_obs_sub_u, (n_p_a * n_cells));
for i_cell = 1 : n_cells
    padded_spk_ts = [zeros(n_p_a, 1); spk_ts_hist(1 : (end - 1), i_cell)];
    x_spk_h_all(:, ((i_cell - 1) * n_p_a + 1) : (i_cell * n_p_a)) = ...
        hankel(padded_spk_ts(1 : (end - n_p_a + 1)), ...
               padded_spk_ts((end - n_p_a + 1) : end));
end
% Prepend stim filter params to get final design matrix.
padded_stim = [zeros(n_p_s - 1, 1); stim_sub_u];
x_stim = hankel(padded_stim(1 : (end - n_p_s + 1)), ...
                padded_stim((end - n_p_s + 1) : end));
x = [x_stim, x_spk_h_all];

% Visualize a small portion of the design matrix.
clf
n_obs_disp = ceil(1 / dt_u);
subplot(1, 10, 1:9);
imagesc(1 : (n_p_s + (n_p_a * n_cells)), 1 : n_obs_disp, ...
        x(1 : n_obs_disp, :));
colormap(gray)
h_cb = colorbar;
h_cb.Location = 'southoutside';
h_cb.Label.String = 'stim intensity value or spike count';
set(gca, 'xticklabel', []);
ylabel('time bin of response');
title('design matrix (stim & cell spike history)');
subplot(1, 10, 10); 
imagesc(y(1 : n_obs_disp, cell_num));
h_cb = colorbar;
h_cb.Location = 'southoutside';
set(gca, 'yticklabel', []);
set(gca, 'xticklabel', []);
title('spike count');

% Divide the complete dataset into a 60:20:20 training:validation:test 
% ratio, and pick random observations for each subset based on this ratio.
obs_all = [1 : n_obs_sub_u]';
n_obs_train = ceil(.6 * n_obs_sub_u);
n_obs_validate = (n_obs_sub_u - n_obs_train) / 2;
n_obs_test = n_obs_validate;
% Throw error if sum of n_obs in subsets doesn't add up to `n_obs`.
assert((n_obs_train + n_obs_validate + n_obs_test) == n_obs_sub_u, ...
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

%% 4. Fit Poisson GLM (P-GLM) without regularization

% <s Fit model

% Here we'll compute the MLE using `fminunc` instead of `fitglm`.

% Append a vector of ones to design matrix for constant term.
x_train_2 = [ones(n_obs_train, 1), x_train];
x_validate_2 = [ones(n_obs_validate, 1), x_validate];
% Compute spike-triggered-average (STA) as initial value for `fminunc`.
sta = (x_train_2' * y_train(:, cell_num)) ./ sum(y_train(:, cell_num));
% Set `fmminunc` options 
% (OLD VERSION commented out below)
% opts = optimset('Gradobj','on','Hessian','on','display','iter');
opts = ...
    optimoptions('fminunc', 'algorithm', 'trust-region', ...
                 'SpecifyObjectiveGradient', true, ...
                 'HessianFcn', 'objective', 'display', 'iter');
% Set negative log-likelihood as cost function.
cost_fn = @(p) neglogli_poissGLM(p, x_train_2, y_train(:, cell_num), dt_u);
% Run optimization.
fmu_p = fminunc(cost_fn, sta, opts);  % fminunc parameters
% Set the intercept term of `fmu_p` to its negative ^^why??^^
fmu_p(1) = -fmu_p(1);
% Compute model's predicted output for training and validation sets.
fmu_y_train = exp(x_train_2 * fmu_p);
fmu_y_validate = exp(x_validate_2 * fmu_p);
% /s>

% Compare `fminunc` results to `fitglm`.
cp_g_l_m = ...
    fitglm(x_train_2, y_train(:, cell_num), 'distribution', 'poisson',...
           'link', 'log', 'intercept', false);
cp_g_l_m_p_train = cp_g_l_m.Coefficients.Estimate;
cp_g_l_m_y_train = cp_g_l_m.Fitted.Response;
cp_g_l_m_y_validate = exp(x_validate_2 * cp_g_l_m_p_train);

% <s Evaluate model

% <ss Compute r-squared values for the model.

% Compute residuals around mean for training and validation sets.
res_train = mean((y_train(:, cell_num) - mean(y_train(:, cell_num))) .^ 2);
res_validate = mean((y_validate(:, cell_num) ...
               - mean(y_validate(:, cell_num))) .^ 2);

           
cp_g_l_m_m_s_e_train = mean((y_train(:, cell_num) - cp_g_l_m_y_train) .^ 2);
% Compute mean-squared error for fmu model on training set.
fmu_m_s_e_train = mean((y_train(:, cell_num) - fmu_y_train) .^ 2);
% Compute r-squared for P-GLM on training set.
fmu_r2_train = 1 - (fmu_m_s_e_train / res_train);
fmt_r2 = ' of the variance in the data is explained by the model.\n';
fprintf(['Training perf (R^2): P-GLM: %.3f', fmt_r2], fmu_r2_train);

% Compute mean-squared error for fmu model on validation set.
fmu_m_s_e_validate = ...
    mean((y_validate(:, 3) - fmu_y_validate) .^ 2);
% Compute r-squared for P-GLM on validation set.
p_g_l_m_r2_validate = 1 - (fmu_m_s_e_validate / res_validate);
fprintf(['Validation perf (R^2): P-GLM: %.3f', fmt_r2], ...
        p_g_l_m_r2_validate);

% Here we see that the amount of the variance in the training data
% explained by the model is 20% greater than the amount explained in the
% validation data, which suggests that our model may be overfit to the
% training data. We'll now explore regularization to try and reduce the
% difference in the performance on the training vs. validation sets.

%% 5. Fit P-GLM with ridge regression prior

% Ridge regression decreases the fit to the training data by adding a
% penalty (the ridge penalty) to the cost function: this penalty is equal
% to `lambda * sum(w.^2)`, where `lambda` is a constant known as the
% "ridge" parameter.  This is also known as an "L2 penalty".
%
% Minimizing error plus the ridge penalty ("penalized least squares") is
% equivalent to computing the MAP estimate under an iid Gaussian prior on
% the filter coefficients. We penalize the parameter estimates in this way
% because we hope by doing so that we will reduce the variance of our MLEs,
% which will hopefully allow for better generalization (and therefore
% better performance) on the validation set.
%
% To set lambda, we'll try a bunch of possible values and use
% cross-validation to select which is best.

% Set up vector of lambda values (ridge parameters)
lambda_vals = 2 .^ (0 : 10);  % it's common to use log-spaced values
n_l_v = length(lambda_vals);

% Precompute some quantities (X'X and X'*y) for training and test data
i_mat = eye(length(fmu_p));  % identity matrix of size == number of params
i_mat(1,1) = 0;              % remove penalty on intercept term

% Allocate space for train and test errors
neg_lklhood_val_train = zeros(n_l_v, 1);     % training error
neg_lklhood_val_validate = zeros(n_l_v, 1);  % test error
w_ridge = zeros(length(fmu_p), n_l_v);       % param weights for lambdas

% Define train and test log-likelihood functions
neg_lklhood_train_fn = ...
    @(p) neglogli_poissGLM(p, x_train_2, y_train, dt_u); 
neg_lklhood_validate_fn = ...
    @(p) neglogli_poissGLM(p, x_validate_2, y_validate, dt_u); 

% Now compute MAP estimate for each ridge parameter
wmap = fmu_p;  % initialize parameter estimate
clf; plot((wmap * 0), 'k'); hold on; % initialize plot
for i_l_v = 1 : n_l_v
    
    % Compute ridge-penalized MAP estimate
    Cinv = lambda_vals(i_l_v)*i_mat;  % inverse prior covariance
    cost_fn = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    wmap = fminunc(cost_fn,wmap,opts);
    
    % Compute negative logli
    neg_lklhood_val_train(i_l_v) = neg_lklhood_train_fn(wmap);
    neg_lklhood_val_validate(i_l_v) = neg_lklhood_validate_fn(wmap);
    
    % store the filter
    w_ridge(:,i_l_v) = wmap;
    
    % plot it
    plot(ttk,wmap(2:end),'linewidth', 2); 
    title(['ridge estimate: lambda = ', num2str(lambda_vals(i_l_v))]);
    xlabel('time before spike (s)'); drawnow; pause(0.5);
 
end
hold off;
% note that the esimate "shrinks" down as we increase lambda

%% Plot filter estimates and errors for ridge estimates

subplot(222);
plot(ttk,w_ridge(2:end,:)); axis tight;  
title('all ridge estimates');
subplot(221);
semilogx(lambda_vals,-neg_lklhood_val_train,'o-', 'linewidth', 2);
title('training logli');
subplot(223); 
semilogx(lambda_vals,-neg_lklhood_val_validate,'-o', 'linewidth', 2);
xlabel('lambda');
title('test logli');

% Notice that training error gets monotonically worse as we increase lambda
% However, test error has an dip at some optimal, intermediate value.

% Determine which lambda is best by selecting one with lowest test error 
[~,imin] = min(neg_lklhood_val_validate);
filt_ridge= w_ridge(2:end,imin);
subplot(224);
plot(ttk,ttk*0, 'k--', ttk,filt_ridge,'linewidth', 2);
xlabel('time before spike (s)'); axis tight;
title('best ridge estimate');


%% === 6. L2 smoothing prior ===========================

% Use penalty on the squared differences between filter coefficients,
% penalizing large jumps between successive filter elements. This is
% equivalent to placing an iid zero-mean Gaussian prior on the increments
% between filter coeffs.  (See tutorial 3 for visualization of the prior
% covariance).

% This matrix computes differences between adjacent coeffs
Dx1 = spdiags(ones(ntfilt,1)*[-1 1],0:1,ntfilt-1,ntfilt); 
Dx = Dx1'*Dx1; % computes squared diffs

% Select smoothing penalty by cross-validation 
lambda_vals = 2.^(1:14); % grid of lambda values (ridge parameters)
n_l_v = length(lambda_vals);

% Embed Dx matrix in matrix with one extra row/column for constant coeff
D = blkdiag(0,Dx); 

% Allocate space for train and test errors
negLtrain_sm = zeros(n_l_v,1);  % training error
negLtest_sm = zeros(n_l_v,1);   % test error
w_smooth = zeros(ntfilt+1,n_l_v); % filters for each lambda

% Now compute MAP estimate for each ridge parameter
clf; plot(ttk,ttk*0,'k'); hold on; % initialize plot
wmap = fmu_p; % initialize with ML fit
for i_l_v = 1:n_l_v
    
    % Compute MAP estimate
    Cinv = lambda_vals(i_l_v)*D; % set inverse prior covariance
    cost_fn = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    wmap = fminunc(cost_fn,wmap,opts);
    
    % Compute negative logli
    negLtrain_sm(i_l_v) = negLtrainfun(wmap); % training loss
    negLtest_sm(i_l_v) = neg_lklhood_validate_fn(wmap); % test loss
    
    % store the filter
    w_smooth(:,i_l_v) = wmap;
    
    % plot it
    plot(ttk,wmap(2:end),'linewidth',2);
    title(['smoothing estimate: lambda = ', num2str(lambda_vals(i_l_v))]);
    xlabel('time before spike (s)'); drawnow; pause(.5);
 
end
hold off;

%% Plot filter estimates and errors for smoothing estimates

subplot(222);
plot(ttk,w_smooth(2:end,:)); axis tight;  
title('all smoothing estimates');
subplot(221);
semilogx(lambda_vals,-negLtrain_sm,'o-', 'linewidth', 2);
title('training LL');
subplot(223); 
semilogx(lambda_vals,-negLtest_sm,'-o', 'linewidth', 2);
xlabel('lambda');
title('test LL');

% Notice that training error gets monotonically worse as 5we increase lambda
% However, test error has an dip at some optimal, intermediate value.

% Determine which lambda is best by selecting one with lowest test error 
[~,imin] = min(negLtest_sm);
filt_smooth= w_smooth(2:end,imin);
subplot(224);
h = plot(ttk,ttk*0, 'k--', ttk,filt_ridge,...
    ttk,filt_smooth,'linewidth', 1);
xlabel('time before spike (s)'); axis tight;
title('best smoothing estimate');
legend(h(2:3), 'ridge', 'L2 smoothing', 'location', 'northwest');
% clearly the "L2 smoothing" filter looks better by eye!

% Last, lets see which one actually achieved lower test error
fprintf('\nBest ridge test LL:      %.5f\n', -min(neg_lklhood_val_validate));
fprintf('Best smoothing test LL:  %.5f\n', -min(negLtest_sm));


%% Advanced exercise:
% --------------------
%
% 1. Repeat of the above, but incorporate spike history filters as in
% tutorial2. Use a different smoothing hyperparamter for the spike-history
% / coupling filters than for the stim filter. In this case one needs to
% build a block diagonal prior covariance, with one block for each group of
% coefficients.
 