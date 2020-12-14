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

%% 4. Fit Poisson GLM w/o regularization

% <s Fit model

% Here we'll compute the MLE using `fminunc` instead of `fitglm`.

% Add a vector of ones to design matrix as constant term.
x_train_2 = [-ones(n_obs_train, 1), x_train];
x_validate_2 = [-ones(n_obs_validate, 1), x_validate];
% Compute spike-triggered-average (STA) as initial value for `fminunc`.
sta = (x_train' * y_train(:, cell_num)) ./ sum(y_train(:, cell_num));
% Set `fmminunc` options 
% (OLD VERSION commented out below)
% opts = optimset('Gradobj','on','Hessian','on','display','iter');
opts = ...
    optimoptions('fminunc', 'algorithm', 'trust-region', ...
                 'SpecifyObjectiveGradient', true, ...
                 'HessianFcn', 'objective', 'display', 'iter');
% Set negative log-likelihood as cost function.
cost_fn = @(p) neglogli_poissGLM(p, x_train, y_train(:, cell_num), dt_u);
% Run optimization.
fmu_p = fminunc(cost_fn, sta, opts);  % fminunc parameters
% Compute model's predicted output
fmu_y_train = exp(x_train * fmu_p);
cp_g_l_m_y_train = exp(x_train_2 * cp_g_l_m_p_train);
fmu_y_validate = exp(x_validate_2 * fmu_p);
% /s>

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

ttk = (-ntfilt+1:0)*dt_u;
h = plot(ttk,ttk*0,'k', ttk,fmu_p(2:end)); 
set(h(2), 'linewidth',2); axis tight;
xlabel('time before spike'); ylabel('coefficient');
title('Maximum likelihood filter estimate'); 

% Looks bad due to lack of regularization!

%% === 5. Ridge regression prior ======================

% Now let's regularize by adding a penalty on the sum of squared filter
% coefficients w(i) of the form:   
%       penalty(lambda) = lambda*(sum_i w(i).^2),
% where lambda is known as the "ridge" parameter.  As noted in tutorial3,
% this is equivalent to placing an iid zero-mean Gaussian prior on the RF
% coefficients with variance equal to 1/lambda. Lambda is thus the inverse
% variance or "precision" of the prior.

% To set lambda, we'll try a grid of values and use
% cross-validation (test error) to select which is best.  

% Set up grid of lambda values (ridge parameters)
lamvals = 2.^(0:10); % it's common to use a log-spaced set of values
nlam = length(lamvals);

% Precompute some quantities (X'X and X'*y) for training and test data
Imat = eye(ntfilt+1); % identity matrix of size of filter + const
Imat(1,1) = 0; % remove penalty on constant dc offset

% Allocate space for train and test errors
negLtrain = zeros(nlam,1);  % training error
negLtest = zeros(nlam,1);   % test error
w_ridge = zeros(ntfilt+1,nlam); % filters for each lambda

% Define train and test log-likelihood funcs
negLtrainfun = @(prs)neglogli_poissGLM(prs,Xtrain,spstrain,dt_u); 
negLtestfun = @(prs)neglogli_poissGLM(prs,Xtest,spstest,dt_u); 

% Now compute MAP estimate for each ridge parameter
wmap = fmu_p; % initialize parameter estimate
clf; plot(ttk,ttk*0,'k'); hold on; % initialize plot
for jj = 1:nlam
    
    % Compute ridge-penalized MAP estimate
    Cinv = lamvals(jj)*Imat; % set inverse prior covariance
    cost_fn = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    wmap = fminunc(cost_fn,wmap,opts);
    
    % Compute negative logli
    negLtrain(jj) = negLtrainfun(wmap); % training loss
    negLtest(jj) = negLtestfun(wmap); % test loss
    
    % store the filter
    w_ridge(:,jj) = wmap;
    
    % plot it
    plot(ttk,wmap(2:end),'linewidth', 2); 
    title(['ridge estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; pause(0.5);
 
end
hold off;
% note that the esimate "shrinks" down as we increase lambda

%% Plot filter estimates and errors for ridge estimates

subplot(222);
plot(ttk,w_ridge(2:end,:)); axis tight;  
title('all ridge estimates');
subplot(221);
semilogx(lamvals,-negLtrain,'o-', 'linewidth', 2);
title('training logli');
subplot(223); 
semilogx(lamvals,-negLtest,'-o', 'linewidth', 2);
xlabel('lambda');
title('test logli');

% Notice that training error gets monotonically worse as we increase lambda
% However, test error has an dip at some optimal, intermediate value.

% Determine which lambda is best by selecting one with lowest test error 
[~,imin] = min(negLtest);
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
lamvals = 2.^(1:14); % grid of lambda values (ridge parameters)
nlam = length(lamvals);

% Embed Dx matrix in matrix with one extra row/column for constant coeff
D = blkdiag(0,Dx); 

% Allocate space for train and test errors
negLtrain_sm = zeros(nlam,1);  % training error
negLtest_sm = zeros(nlam,1);   % test error
w_smooth = zeros(ntfilt+1,nlam); % filters for each lambda

% Now compute MAP estimate for each ridge parameter
clf; plot(ttk,ttk*0,'k'); hold on; % initialize plot
wmap = fmu_p; % initialize with ML fit
for jj = 1:nlam
    
    % Compute MAP estimate
    Cinv = lamvals(jj)*D; % set inverse prior covariance
    cost_fn = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    wmap = fminunc(cost_fn,wmap,opts);
    
    % Compute negative logli
    negLtrain_sm(jj) = negLtrainfun(wmap); % training loss
    negLtest_sm(jj) = negLtestfun(wmap); % test loss
    
    % store the filter
    w_smooth(:,jj) = wmap;
    
    % plot it
    plot(ttk,wmap(2:end),'linewidth',2);
    title(['smoothing estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; pause(.5);
 
end
hold off;

%% Plot filter estimates and errors for smoothing estimates

subplot(222);
plot(ttk,w_smooth(2:end,:)); axis tight;  
title('all smoothing estimates');
subplot(221);
semilogx(lamvals,-negLtrain_sm,'o-', 'linewidth', 2);
title('training LL');
subplot(223); 
semilogx(lamvals,-negLtest_sm,'-o', 'linewidth', 2);
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
fprintf('\nBest ridge test LL:      %.5f\n', -min(negLtest));
fprintf('Best smoothing test LL:  %.5f\n', -min(negLtest_sm));


%% Advanced exercise:
% --------------------
%
% 1. Repeat of the above, but incorporate spike history filters as in
% tutorial2. Use a different smoothing hyperparamter for the spike-history
% / coupling filters than for the stim filter. In this case one needs to
% build a block diagonal prior covariance, with one block for each group of
% coefficients.
 