% tutorial1_PoissonGLM.m
%
% This tutorial covers using generalized linear models (GLMs) to model
% neural spiking data. This tutorial assumes basic-to-intermediate
% familiariy with behavioral neuroscience experiments, neural spike data,
% statistics / statistical distributions, linear algebra, and
% MATLAB programming.
%
% This tutorial illustrates the fitting of a Gaussian GLM (aka 
% linear-least-squares regression) and a Poisson GLM (aka 
% linear-nonlinear-Poisson regression) to retinal ganglion cell spike
% trains that are stimulated with visual presentation of binary white noise
% (the normalized intensity of the white noise - the relative ratio of 
% black to white pixels on a gray screen - changes between two distinct 
% values over time).
%
% The purpose of using GLMs in this context is to predict the response
% variable - the spiking activity in either rate or binned counts - from 
% predictor variables - the past intensity values of the stimulus.
%
% DATASET: this tutorial is designed to run with retinal ganglion cell
% spike train data from Uzzell & Chichilnisky 2004. The dataset can be 
% made available upon request to the authors.

%% How to use this tutorial:
%
% This is an interactive tutorial designed to walk you through the steps of
% fitting two classic models (linear-Gaussian GLM and Poisson GLM) to spike
% train data. It is organized into 'sections'.
%
% In the following, I recommend positioning the figure window (once it
% appears) in a place where you can easily see it (e.g., 'docked' beside or
% above the editor / command windows, or else in its own place on the
% screen), with no other matlab windows you need on top of it or underneath
% it. Each section of the tutorial will overwrite the figure window. The 
% figure window will always display the plots made in the current section,
% so there's no need to go clicking through multiple windows to find the 
% one you're looking for!
%
% Be sure to place the data file on your MATLAB path before running the
% tutorial.

%% 1. Load and plot the raw data

% Load data.
data_dir = 'data_RGCs/';       % data directory
load([data_dir, 'Stim']);      % stimulus values (binary white noise)
load([data_dir,'stimtimes']);  % stim frame times (in s)
load([data_dir, 'SpTimes']);   % spike times for 4 cells (in s)

% Rename variables.
stim = Stim;
stim_ts = stimtimes;
spk_ts = SpTimes;
clear Stim stimtimes SpTimes

% Pick a cell to work with.
cell_num = 3; % (1-2 are OFF cells; 3-4 are ON cells).
spk_ts_cell = spk_ts{cell_num};

% Compute some basic statistics on the data.
dt = (stim_ts(2) - stim_ts(1));  % time bin size for stim (s)
n_obs = size(stim, 1);           % number of stim samples (observations)
n_spks = length(spk_ts_cell);    % number of spikes

% Print out some basic info.
fprintf('--------------------------\n');
fprintf('Loaded RGC data: cell %d\n', cell_num);
fprintf('Number of stim frames: %d (%.1f minutes)\n', n_obs, ...
        n_obs * (dt / 60));
fprintf('Time bin size: %.5f s\n', dt);
fprintf('Number of spikes: %d (mean rate = %.1f Hz)\n\n', n_spks, ...
        n_spks / n_obs * (1 / dt));

% Plot first second of stimulus.
figure
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
    spk_ts_cell((spk_ts_cell >= t_in_stim_bins_1s(1)) ...
                     & (spk_ts_cell < t_in_stim_bins_1s(end)));
plot(spk_ts_in_1s, 1, 'ko', 'markerfacecolor', 'k');
axis tight
set(gca, 'xlim', t_in_stim_bins_1s([1 end]));
title('spike times');
xlabel('time (s)');

%% 2. Create the response variable: bin the spike train

% For now, we will assume we want to use the same bins on the spike train
% as the bins used for the stimulus. Later, though, we'll wish to vary 
% this. This binned spike activity vector (where we have a spike count for
% each observation) will be what we are trying to predict from our GLMs.

% Create time bins for spike train binning.
spk_ts_bins = [0, ((1 : n_obs) * dt)];
% Bin the spike train.
spk_ts_hist = histcounts(spk_ts_cell, spk_ts_bins);
% Ensure column vector and rename as `y`.
y = spk_ts_hist(:);

% Replot spiking data as counts.
subplot(2, 1, 2);
stem(t_in_stim_bins_1s, y(stim_bins_1s), 'k', 'linewidth', 2);
title('binned spike counts');
ylabel('spike count');
xlabel('time (s)');
axis tight

%% 3. Create the predictor variables: build the design matrix

% This is a necessary step before we can fit the model: we need to 
% assemble a matrix that contains the relevant predictor variables 
% (aka regressors aka covariates aka parameters) for each time bin of the 
% response. This is known as a design matrix. Each row of this matrix
% contains the relevant past stimulus chunk for predicting the spike count 
% at a given time bin.

% Set the number of past bins of the stimulus to use to predict spikes.
% (Try varying this to see how performance changes!)
n_p_x = 25;  % number of parameters in `x` design matrix (bins in past)
% Pad early bins of stimulus with zero.
padded_stim = [zeros(n_p_x - 1, 1); stim];

% Preallocate and build design matrix row by row.
x = zeros(n_obs, n_p_x);
for i_obs = 1 : n_obs
    x(i_obs, :) = padded_stim(i_obs : (i_obs + n_p_x - 1));
end

% There's actually a faster and more elegant way to build the design 
% matrix. The design matrix here is known as a Hankel matrix, so we can 
% build our design matrix as a Hankel matrix. A Hankel matrix is entirely
% determined by its first column and last row. 
% (Type 'help hankel' to learn more).
x = hankel(padded_stim(1 : (end - n_p_x + 1)), ...
           padded_stim((end - n_p_x + 1) : end));

% Let's visualize a small part of the design matrix just to see it.
clf
n_obs_disp = 100;  % number of observations to display
% Display an `n_r_x` by `n_obs_disp` portion of `x`.
imagesc(-n_p_x + 1 : 0, 1 : n_obs_disp, x(1 : n_obs_disp, :));
h_cb = colorbar;
colormap(gray)
h_cb.Label.String = 'stim intensity values';
xlabel('lags before spike time bin');
ylabel('time bin of response');
title('design matrix');

% Notice that the design matrix has a structure where every row is a 
% shifted copy of the row above, which comes from the fact that for each 
% time bin of response, we're grabbing the preceding `n_p_x` bins of 
% stimulus as the predictors.

% We now need to partition our data into 3 sets: a "training set", a
% "validation" set, and a "test/holdout" set. We will use the training set 
% to fit our models, we will use the validation set to see the performance
% of our models after the training (and we will use this performance to
% decide whether we want to adjust any hyperparameters of our models and
% re-train), and when we have made final adjustments to the models, we will
% use the test set to measure the performance of our final models. 
% We use these divisions in order to prevent "overfitting": where a model
% performs well on a training and/or validation set but fails to generalize
% to the validation and/or test sets.

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
y_train = y(obs_train);
y_validate = y(obs_validate, :);
y_test = y(obs_test, :);

%% 4. The core features of a GLM

% Now that we have our design matrix, we can start building and testing
% GLMs to see how well they predict the spike data.
%
% In order to understand the usefulness of GLMs, let's first consider
% standard 2-d normal linear regression on an x-y axis. In particular,
% let's consider the curve (which in this case will be a line) of best fit
% through the data. The curve of best fit will be the line that minimizes
% the mean squared error between itself and our empirical y-values, out of
% all possible curves. An unusual way we can think of this curve is as
% estimating the mean values of normal distributions (the y-values of the
% curve) for each unique x-value. In other words, we can think of each
% empirical y-value as drawn from a separate normal distribution whose mean
% is equal to the y-value of the line of best fit at the corresponding 
% x-value. We can write this in equation form as the classic `y = ax + b`.
%
% Now, imagine our response variable `y` clearly does not depend on `x` in
% a linear relationship. Instead of thinking about each empirical y-value
% as being drawn from a normal distribution, we can think of them as being
% drawn from some other distribution, e.g. a Poisson distribution. So, each
% y-value of our curve of best fit will be the mean value of a Poisson
% distribution, parametrized by the corresponding x-value. It turns out
% that if we can think of our response variable as being drawn from any
% distribution within the exponential family of distributions, then we can
% think of each point on the curve of best fit as estimating the mean value
% of this distribution, parametrized by the corresponding predictors, and 
% we can use a GLM to find these mean values, or i.e. the curve of best 
% fit.
%
% So, how does a GLM find the mean values of the specified response
% variable distribution that will best fit the data?
%
% **A GLM treats these mean values as the output of a function that
% operates on a linear combination of predictor variables
% `E(y) = f((B1 * x1) + (B2 * x2) + ... (Bn * xn))`
% (where `E(y)` represents the mean values that will be output from the 
% "mean function", `f`, that operates on a linear combination of predictor
% variables (the `x` terms), that each have an associated weight (the `B`
% terms)).
% and finds the optimal values for the `B` weights such that _the joint
% probability of the observed data as a function of the unknown parameters
% for the chosen response variable distribution_ (known as the "likelihood
% function"), is maximized.** 

% This concept of maximum likelihood estimation (MLE) can be thought of as 
% roughly equivalent to minimizing the mean squared error (or another error
% function) between the resulting `E(y)` values and the empirical response
% variable data. 
% For an introduction to MLE, see this article:
% https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1
%
% One way in which a GLM actually searches the parameter space to find
% the best-fit parameters is via an optimization algorithm, e.g. gradient
% descent (GD) on the negative likelihood function (minimizing the negative
% likelihood function is equivalent to maximizing the likelihood function). 
% For an introduction to GD, see this article:
% https://towardsdatascience.com/gradient-descent-explained-9b953fc0d2c
%
% (*Note*, the inverse of the "mean function" is called the 
% "link function", which expresses the linear combination of predictor 
% variables as some function that operates on the mean values of the
% response variable distribution, i.e. `f(E(y))`. The term "link function"
% tends to be used more often than "mean function", so it's important to
% understand that they are just inverses of each other.)
%
% The namesake for "G" in GLM comes from the fact that we can "generalize"
% standard linear regression (which assumes the response variable is
% normally distributed) to predict response variables that come from other
% distributions in the exponential family. The "L" comes from the fact that
% the function that predicts the mean values of the response variable must
% operate on only linear combinations of predictor variables: e.g., we
% could use `f((B1 * x1) - (B2 * x2 ^ 2))`, but not `f((B1 * (x1 ^ x2)))`.
% We can use nonlinearities that operate on individual parameters, which is
% why the `(B2 * x2 ^ 2)` term is fine, but they must only be combined
% linearly, which is why the `(x1 ^ x2)` term is not allowed.
%
% So to summarize, a GLM has three components:
%
% 1) The response variable distribution (must belong to the exponential
% family of distributions)
% 2) The linear combination of predictor variables used to predict the mean
% values of the response variable distribution
% 3) The mean function: the function that operates on the linear
% combination of predictor variables.
%
% and the core goal of a GLM is to find the optimal weights for the
% predictor variables such that some error function between the output of
% the mean function and the empirical response variable data is minimized.
%
% (In order to compute the likelihood, there is actually also a fourth
% componenet, the "variance function", which determines how the variance
% (aka scale) depends on the mean (aka shape): `Var(y) = f(E(y))` for the
% response variable distribution. The variance values are not directly used
% in computing the fitted model values, however.)
%
% Our first models will resemble Gaussian GLMs (G-GLMs) with the identity
% link function. The identity link function does not transform the linear 
% combination of predictor variables in any way, hence the namesake 
% "identity". So, we can think of these models as just performing standard
% multiple linear (aka normal aka Gaussian) regression. 
%
% We will then move onto Poisson GLMs (P-GLMs), and we will use the 
% log link function, which transforms the linear combination of predictor
% variables by using them as exponents for base `e`. 
% i.e. `E(y) = exp(B1 * x1 + ... Bn * xn)`
%
% There are multiple ways we can find estimates for the best-fit parameter
% weights, `p`, for a G-GLM:
%
% 1) We can use the spike-triggered average (STA).
% 2) We can use linear algebra and the normal equation.
% 3) We can perform an optimization algorithm on the likelihood function.
%
% We will explain, use, and compare all three of these techniques below.

%% 5a. Predicting spikes with a G-GLM: using the STA

% Our first method of estimating `p` for a G-GLM will use the STA. We 
% can think of the STA as a filter that, when convolved with the `n_p_x`
% preceding stimulus values of an observation, predicts the spike count for
% the observation.
%
% When the stimulus is Gaussian white noise, the STA is an unbiased
% estimator for the best-fit parameters in a G-GLM. ^^why?? and what 
% is meant by, and what happens, when the "nonlinearity results in an STA
% whose expectation is zero" ?^^
%
% In many cases it's useful to visualize the STA regardless of if we're 
% going to use it for data prediction, just because if we don't see any 
% kind of structure in the STA then this may indicate that we have a 
% problem (e.g. a mismatch between the design matrix and binned spike
% counts).

% Now that we have the design matrix, it's easy to compute the STA.
% Remember, we will use the training set for all model fits.
s_t_a = (x_train' * y_train) / n_spks;

% Plot the STA (this should look like a biphasic filter).
clf
s_t_a_bins = (-n_p_x + 1 : 0) * dt;  % time bins for STA (in s)
plot(s_t_a_bins, s_t_a, 'o-', 'linewidth', 2);
axis tight;
title('STA');
xlabel('time before spike (s)');
ylabel('stim intensity');

% If the stimuli are non-white, then the STA is generally a biased
% estimator for the best-fit parameters in a G-GLM ^^(why??)^^. In this
% case, we can compute the whitened STA, which is the MLE for the best-fit
% parameters of a G-GLM.
%
% If the stimuli have correlations, this ML estimate may look like garbage
% (more on this in the tutorial on "regularization"). But for this dataset,
% we know that the stimuli are white, so we don't (in general) expect a big
% difference from the STA.

% First we whiten our design matrix.
w_m_x = chol(inv(cov(x_train)));  % whitening matrix for `x`
w_x = x_train * w_m_x;            % whitened design matrix

% Then compute the whitened STA.
w_s_t_a = w_x' * y_train / n_spks;
% (*Note*: another roughly equivalent way to get the whitened STA is:
% `w_s_t_a = inv(x' * x) * (x' * y)` 
% This is the normal equation: the matrix form of least-squares 
% regression!)

% Let's plot both the `s_t_a` and `w_s_t_a` rescaled as unit vectors (so we
% can see any differences in their shape).
clf
plot(s_t_a_bins, (s_t_a ./ norm(s_t_a)), 'o-', 'linewidth', 2);
hold on
plot(s_t_a_bins, (w_s_t_a ./ norm(w_s_t_a)), 'o-', 'linewidth', 2);
axis tight
title('Unit norm STA and whitened STA'); 
xlabel('time before spike (s)');
ylabel('stim intensity');
legend('STA', 'wSTA', 'location', 'northwest');

% When rescaled as unit vectors, we see that the STA and WSTA
% completely overlap, which confirms that our original data is indeed 
% white.

% Let's visualize the performance of the model for the first second of 
% data, and then quantify the performance on the training and validation
% sets.
% Get model predictions.
g_g_l_m_s_t_a_y_train = x_train * w_s_t_a;
g_g_l_m_s_t_a_y_validate = x_validate * w_s_t_a;
% Plot model prediction on top of empirical data.
clf
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_s_t_a_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM STA');
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_s_t_a_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM STA');
% Get model performance.
% Let's report the mean residuals for our binned spike count and the
% relevant training error (mean squared prediction error) for the model.
res_train = mean((y_train - mean(y_train)) .^ 2);
g_g_l_m_s_t_a_m_s_e_train = mean((y_train - g_g_l_m_s_t_a_y_train) .^ 2);
fprintf('Training perf (R^2): G-GLM STA: %.3f\n', ...
        1 - (g_g_l_m_s_t_a_m_s_e_train / res_train));
res_validate = mean((y_validate - mean(y_validate)) .^ 2);
g_g_l_m_s_t_a_m_s_e_validate = ...
    mean((y_validate - g_g_l_m_s_t_a_y_validate) .^ 2);
fprintf('Validation perf (R^2): G-GLM STA: %.3f\n', ...
        1 - (g_g_l_m_s_t_a_m_s_e_validate / res_validate));

% We can see in the plots that it seems the model does trend upwards every
% time there is a positive spike count, however, the model is far from
% ideal. It seems to consistently under predict the spike count, and allows
% for negative spike counts, which we know are impossible. Furthermore, the
% R^2 values tell us that it only has 10.8% and 10.4% less error than the
% mean residuals in predicting the data for the training and validation
% sets, respectively. It does at least seem that our model hasn't overfit,
% as it performs nearly as well on the validation set as on the training
% set.

%% 5b. Predicting spikes with a G-GLM: using the normal equation

% It turns out that the best-fit parameters for multiple linear regression
% can be perfectly found using the normal equation from linear algebra.
% 
% Let's call the error function that we are trying to minimize `J(p)`, 
% where `p` is a vector of our parameter values, which is what we are 
% trying to solve for. So `J(p) = 1/2 * n * sum((h(x, p) - y) ^ 2), where
% `n` is the number of observations, and `h(x, p)` is the predicted output
% for `x` parametrized by `p`.
%
% To find the values of the parameters that minimize `J(p)`, we should take
% the derivative of our cost function with respect to the parameters, 
% i.e. `dp / dJ(p)`, set it equal to 0, and solve for p. 
%
% When we do so, we find:
% `p = inv(x' * x) * (x' * y)`
% To see this derivation in a nice, short article, see: 
% https://towardsdatascience.com/normal-equation-a-matrix-approach-to-linear-regression-4162ee170243
%
% As mentioned in the previous section, this is the normal equation, the 
% matrix form of least-squares regression.

% Use normal equation (NE) to find params for g-glm.
g_g_l_m_n_e_p = inv(x_train' * x_train) * (x_train' * y_train);

% Plot these parameters as a rescaled unit vector over the STA to compare.
clf
hold on
plot(s_t_a_bins, (w_s_t_a ./ norm(w_s_t_a)), 'o-', 'linewidth', 2);
plot(s_t_a_bins, (g_g_l_m_n_e_p ./ norm(g_g_l_m_n_e_p)), 'o-', ....
     'linewidth', 2);
axis tight
legend('WSTA', 'NE', 'location', 'northwest');
title('Unit norm best-fit params from various methods for G-GLM'); 
xlabel('time before spike (s)');
ylabel('stim intensity');

% When rescaled as unit vectors, we see that the parameters we return after
% solving the normal equation overlap with the WSTA, which confirms that 
% the WSTA is a good estimator for the best-fit parameters in a G-GLM.

% Let's visualize the performance of the model for the first second of 
% data, and then quantify the performance on the training and validation
% sets.
% Get model predictions.
g_g_l_m_n_e_y_train = x_train * g_g_l_m_n_e_p;
g_g_l_m_n_e_y_validate = x_validate * g_g_l_m_n_e_p;
% Plot model prediction on top of previous model and empirical data.
clf
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_s_t_a_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_n_e_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM STA', 'G-GLM NE');
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_s_t_a_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_n_e_y_validate(stim_bins_1s), ...
     'linewidth', 1.5); 
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM STA', 'G-GLM NE');
% Get model performance.
% Let's report the mean residuals for our binned spike count and the
% relevant training error (mean squared prediction error) for the model.
g_g_l_m_n_e_m_s_e_train = mean((y_train - g_g_l_m_n_e_y_train) .^ 2);
fprintf('Training perf (R^2): G-GLM NE: %.3f\n', ...
        1 - (g_g_l_m_n_e_m_s_e_train / res_train));
g_g_l_m_n_e_m_s_e_validate = ...
    mean((y_validate - g_g_l_m_n_e_y_validate) .^ 2);
fprintf('Validation perf (R^2): G-GLM NE: %.3f\n', ...
        1 - (g_g_l_m_n_e_m_s_e_validate / res_validate));

% We can see in the plots and the reported R^2 values that the WSTA
% performs nearly identically to the best-fit parameters for a G-GLM found
% via solving the normal equation. We knew this would be the case when we 
% compared the plot of the WSTA to the G-GLM NE parameters.

%% 5c. Predicting spikes with a G-GLM: using MLE via FS on likelihood

% Arguably the most traditional way to find the best fit parameters for a
% model is to run an optimization algorithm on the likelihood function that
% returns the parameter weights that maximize the likelihood function.
% We could write code to 1) compute the likelihood as a function of our 
% parameters, and 2) run our own optimization algorithm, but for now we
% will use MATLAB's built-in `fitglm`, which uses the "Fisher's Scoring"
% (FS) optimaztion algorithm on the likelihood function.

% Create and fit model.
g_g_l_m = fitglm(x_train, y_train, 'distribution', 'normal', ...
               'link', 'identity', 'intercept', false);
g_g_l_m_p = g_g_l_m.Coefficients.Estimate;

% Plot these params as a rescaled unit vector over the others to compare.
clf
hold on
plot(s_t_a_bins, (w_s_t_a ./ norm(w_s_t_a)), 'o-', 'linewidth', 2);
plot(s_t_a_bins, (g_g_l_m_n_e_p ./ norm(g_g_l_m_n_e_p)), 'o-', ....
     'linewidth', 2);
plot(s_t_a_bins, (g_g_l_m_p ./ norm(g_g_l_m_p)), 'o-', 'linewidth', 2); 
axis tight
title('Unit norm best-fit params from various methods for G-GLM'); 
xlabel('time before spike (s)');
ylabel('stim intensity');
legend('WSTA', 'NE', 'MLE-FS', 'location', 'northwest');

% When rescaled as unit vectors, we see that the parameter estimates
% returned by all of our parameter estimate methods overlap. We therefore
% know that performance for this model will be roughly equivalent to our
% two previous G-GLMs.
%
% One thing we didn't include in our model is a constant / intercept
% parameter that will allow our spike count prediction to have a non-zero
% mean. We can interpret this parameter as the baseline firing rate. There
% are two ways we can account for this:
% 1) We can add a column of ones directly to our design matrix
x_train_2 = [ones(n_obs_train, 1), x_train];
x_validate_2 = [ones(n_obs_validate, 1), x_validate];
% And run `fitglm` on this design matrix.
g_g_l_m = fitglm(x_train_2, y_train, 'distribution', 'normal', ...
                 'link', 'identity', 'intercept', false);
% 2) Or we can tell `fitglm` to include this intercept term via the
% `'intercept'` name-value pair arg
g_g_l_m_2 = fitglm(x_train, y_train, 'distribution', 'normal', ...
                   'link', 'identity', 'intercept', true);
% Ensure the parameter weights for both methods above are equal.
g_g_l_m_p = g_g_l_m.Coefficients.Estimate;
g_g_l_m_p_2 = g_g_l_m_2.Coefficients.Estimate;
assert(all(round(g_g_l_m_p, 5) == round(g_g_l_m_p_2, 5)));

% Let's now visualize and quantify the performance of this G-GLM (with an
% intercept parameter) to the previous model.
% Get model predictions.
g_g_l_m_y_train = x_train_2 * g_g_l_m_p;
g_g_l_m_y_validate = x_validate_2 * g_g_l_m_p;
% Plot model prediction on top of previous model and empirical data.
clf
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_n_e_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM NE', 'G-GLM MLE-FS');
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_n_e_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5); 
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM NE', 'G-GLM MLE-FS');
% Get model performance.
% Let's report the mean residuals for our binned spike count and the
% relevant training error (mean squared prediction error) for the model.
g_g_l_m_m_s_e_train = mean((y_train - g_g_l_m_y_train) .^ 2);
fprintf('Training perf (R^2): G-GLM MLE-FS: %.3f\n', ...
        1 - (g_g_l_m_m_s_e_train / res_train));
g_g_l_m_m_s_e_validate = ...
    mean((y_validate - g_g_l_m_y_validate) .^ 2);
fprintf('Validation perf (R^2): G-GLM MLE-FS: %.3f\n', ...
        1 - (g_g_l_m_m_s_e_validate / res_validate));

% We can see visually that this version of the model with an intercept term
% fits the data better than G-GLMs without the intercept term, and the R^2
% values increase to over .38 for both the training and validation sets.
% However, this model still seems to undercount the data and still suffers
% from outputting impossible negative spike counts. Since this latest G-GLM
% is our best G-GLM, going forward when we refer to the G-GLM we will be
% referring to this model.

%% 6. Fitting & predicting with a Poisson GLM

% In order to try and fix the issues that plagued our Gaussian GLM (namely,
% outputting negative spike counts and undercounting the empirical data),
% let's finally move on to constructing a Poisson GLM with a log link!

% Use `fitglm` to construct the model.
p_g_l_m = fitglm(x_train_2, y_train, 'distribution', 'poisson', ...
                 'link', 'log', 'intercept', false);

% Compute prediction of model's spike count values via the mean function. 
% Because we are using a log link function, the mean function is the
% inverse, `e^(...)`, or in MATLAB, `exp`.
p_g_l_m_p = p_g_l_m.Coefficients.Estimate;
p_g_l_m_y_train = exp(x_train_2 * p_g_l_m_p);
p_g_l_m_y_validate = exp(x_validate_2 * p_g_l_m_p);
% The predicted values are also returned directly in the glm object:
p_g_l_m_y_train_2 = p_g_l_m.Fitted.Response;
% Ensure model values for both methods above are equal.
assert(all(round(p_g_l_m_y_train, 5) == round(p_g_l_m_y_train_2, 5)))
% Visually and quantitatively compare the P-GLM to the best G-GLM.
clf
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM', 'P-GLM');
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM', 'P-GLM');
% Report training performance.
p_g_l_m_m_s_e_train = mean((y_train - p_g_l_m_y_train) .^ 2);
fprintf('Training perf (R^2): P-GLM: %.3f\n', ...
        1 - (p_g_l_m_m_s_e_train / res_train));
p_g_l_m_m_s_e_validate = ...
    mean((y_validate - p_g_l_m_y_validate) .^ 2);
fprintf('Validation perf (R^2): P-GLM: %.3f\n', ...
        1 - (p_g_l_m_m_s_e_validate / res_validate));
    
% Here we see that the Poisson GLM is the best fit to our data out of the
% three models we have looked at so far, yet it still only has an R^2 value
% of only about 0.5 on both the test and validation sets.

%% 7. Non-parametric estimate of the nonlinearity

% The above P-GLM used an exponential nonlinearity (the mapping from filter
% output (the best-fit parameters) to spike count). We can also use a 
% "nonparametric" estimate of the nonlinearity using a more flexible
% class of functions.
%
% Let's use the family of piecewise constant functions to predict the
% spiking activity. This can be done via a simple estimation procedure:
% 1. Compute the raw predicted output values from the Poisson GLM's
% parameters (these output values are the ones that are computed before
% getting passed through the mean function).
% 2. Bin the raw predicted output values (where the number of bins = the
% number of parameters).
% 3. In each bin, compute the fraction of stimuli-elicted spikes. This
% value will be that bin's (parameter's) weight for the corresponding raw
% predicted output value for the nonparametric GLM (NP-GLM).

% Initialize the parameters array for the NP-GLM.
np_g_l_m_p = zeros(n_p_x, 1);
% Get P-GLM raw predicted output values.
p_g_l_m_y_raw_train = x_train_2 * p_g_l_m_p;
p_g_l_m_y_raw_validate = x_validate_2 * p_g_l_m_p;
% The raw predicted values are also returned directly in the glm object:
p_g_l_m_y_raw_train_2 = p_g_l_m.Fitted.LinearPredictor;
% Ensure raw predicted values for both methods above are equal.
assert(all(round(p_g_l_m_y_raw_train, 5) ...
           == round(p_g_l_m_y_raw_train_2, 5)));
% Bin the raw predicted output values and get the bin edges and the bin
% index of each observation in the raw predicted output.
[p_g_l_m_y_raw_hist, bin_edges, bin_idxs] = ...
    histcounts(p_g_l_m_y_raw_train, n_p_x);
% Print the bin edges to the screen.
fmt = ['Bin edges for the histogram of the raw predicted output are: \n' ...
       repmat(' %.2f', 1, numel(bin_edges))];
fprintf(fmt, bin_edges);
fprintf('\n');
% Compute mean spike count in each bin, and assign to the parameters array.
for i_bin = 1 : n_p_x
    np_g_l_m_p(i_bin) = mean(y_train(bin_idxs == i_bin));
end

% Predict values for NP-GLM.
% Create an array of values at the bin centers for plotting / prediction.
x_bin_cntrs = bin_edges(1 : (end - 1)) + (diff(bin_edges(1:2)) / 2);
% Now let's embed this in a function we can evaluate at any value in the
% raw predicted output values. This function will be the equivalent of the
% mean function.
np_g_l_m_f = ...
    @(xq) interp1(x_bin_cntrs, np_g_l_m_p, xq, 'nearest', 'extrap');
% And let's use this function on the training and validation sets.
np_g_l_m_y_train = np_g_l_m_f(p_g_l_m_y_raw_train);
np_g_l_m_y_validate = np_g_l_m_f(p_g_l_m_y_raw_validate);

% Make plots: 1) histogram of raw predicted output values, 2) NP-GLM
% model predictions for the values in the raw predicted output
clf
subplot(2, 1, 1);
bar(x_bin_cntrs, p_g_l_m_y_raw_hist, 'hist');
ylabel('count');
xlabel('raw predicted output value');
title('histogram of raw predicted output values');
axis tight
subplot(2, 1, 2);
% Get x values at which to evaluate `np_g_l_m_f`.
x_raw_p_g_l_m = bin_edges(1) : .01 : bin_edges(end);
% Plot `npf_glm` at `x_raw_p_glm`.
plot(x_raw_p_g_l_m, np_g_l_m_f(x_raw_p_g_l_m), 'linewidth', 1.5);
% Overlay Poisson GLM predictions at `x_raw_p_glm`.
hold on
plot(x_raw_p_g_l_m, exp(x_raw_p_g_l_m), 'linewidth', 1.5);
xlabel('raw predicted output value');
ylabel('spike count');
legend('NPF', 'P-GLM');
title('comparison of NPF to P-GLM');
axis tight

% As we can see from the plots, the NPF model's predicted spike count per
% `dt` bin tapers off under 3 for the highest raw predicted raw output
% values, while the P-GLM predicts over 8 spikes per `dt` bin for these raw
% output values. 

% Let's visualize and quantify this model's performance compared to the
% P-GLM and G-GLM.
clf
subplot(2, 1, 1)
hold on
stem(t_in_stim_bins_1s, y_train(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, np_g_l_m_y_train(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of training data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM', 'P-GLM', 'NP-GLM');
subplot(2, 1, 2)
hold on
stem(t_in_stim_bins_1s, y_validate(stim_bins_1s), 'linewidth', 1.5);
plot(t_in_stim_bins_1s, g_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, p_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
plot(t_in_stim_bins_1s, np_g_l_m_y_validate(stim_bins_1s), ...
     'linewidth', 1.5);
axis tight
title('model fits to 1s of validation data');
xlabel('time (s)');
ylabel('spike count');
legend('empirical spike count', 'G-GLM', 'P-GLM', 'NP-GLM');
% Report training performance.
np_g_l_m_m_s_e_train = mean((y_train - np_g_l_m_y_train) .^ 2);
fprintf('Training perf (R^2): NP-GLM: %.3f\n', ...
        1 - (np_g_l_m_m_s_e_train / res_train));
np_g_l_m_m_s_e_validate = ...
    mean((y_validate - np_g_l_m_y_validate) .^ 2);
fprintf('Validation perf (R^2): NP-GLM: %.3f\n', ...
        1 - (np_g_l_m_m_s_e_validate / res_validate));

% Here we see that the NPF GLM is the best fit to our data out of the four
% models we have looked at so far, with an R^2 near 0.6 on both the
% training and validation sets.

% Advanced exercise: write your own function that acts on some parameters
% to estimate the spike count, and then find the best-fit weights for these
% parameters by first defining a likelihood function as a function of these 
% parameters and then running an optimization algorithm on this likelihood
% function. (For example, iteratively descend the negative log-likelihood 
% via gradient descent).

%% 8a. Reviewing model performance: checking significance of parameters

% Checking individual parameters based on CI + wald test
% Residual analysis
% Log-likelihood + AIC analysis
% KS test based on time-rescaling theorem for p-glm
% MLRTs


%% 8b. Reviewing model performance: residual analysis

%% 8c. Reviewing model performance: Log-likelihood values & AIC

% Now that we've defined and trained some models, we should compute the
% log-likelihood values for each model's best-fit to compare performances. 
% The higher the log-likelihood value, the more likely the data is to be 
% from the model (i.e. the better the model fit). ^^how does this relate to
% comparing our previous reporting of mse??^^

% LOG-LIKELIHOOD (this is what `fitglm` maximizes when fitting the GLM):
% --------------
% Let `s` be the empirical spike count in a bin and `r` the predicted spike 
% rate (known as "conditional intensity") in units of spikes/bin, then: 
%
% Gaussian likelihood:  P(s;r) = 1 / (sigma * sqrt(2 * pi)) ...
%                                * exp((-1 / 2) * ((r - s) / sigma) ^ 2)
% Gaussian log-l:       log(P(s;r)) = -log(sigma * sqrt(2 * pi)) ...
%                                     - ((1 / 2) * ((r - s) / sigma) ^ 2)
% Poisson likelihood:   P(s;r) = (r^s * exp(-r)) / s!
% Poisson log-l:        log(P(s;r)) =  (s * log(r) - r) - log(s!)
%
% These formulae are to compute the probability of observing `s` in a
% particular bin (observation) given the `r` for that bin. The total 
% log-likelihood then is the summed log-likelihood over all bins. 
% (Sum of the log of probabilities = log of the product of probabilities).

% Compute log-likelihood value for G-GLM.
sigma = sqrt(g_g_l_m.Dispersion);
g_g_l_m_ll = ...
    sum(-log(sigma * sqrt(2 * pi))...
        - ((1 / 2) .* (((g_g_l_m_y_train - y_train) ./ sigma) .^2)));

% Compute log-likelihood value for P-GLM.
p_g_l_m_ll = y_train' * log(p_g_l_m_y_train) - sum(p_g_l_m_y_train) ...
             - sum(log(factorial(y_train)));

% The log-likelihood values are also returned directly in the glm object;
% ensure the above calculations match those from the model.
assert(round(p_g_l_m_ll, 5) == round(p_g_l_m.LogLikelihood, 5))
assert(round(g_g_l_m_ll, 5, 'significant') == ...
    round(g_g_l_m.LogLikelihood, 5, 'significant'))

% We can also compute a log-likelihood value estimate for the NP-GLM by 
% using the Poisson log-likelihood function on the NP-GLM predicted values.
% To avoid negative infinites in our computation, we will first replace 0
% values predicted by the model
np_g_l_m_y_train_nonzero = np_g_l_m_y_train;
np_g_l_m_y_train_nonzero(np_g_l_m_y_train_nonzero == 0) = ...
    min(p_g_l_m_y_train);
np_g_l_m_ll = y_train' * log(np_g_l_m_y_train_nonzero) ...
              - sum(np_g_l_m_y_train_nonzero) ...
              - sum(log(factorial(y_train)));

% Lastly, compute log-likelihood for a "homogeneous-Poisson" model (HP-GLM)
% that assumes a constant firing rate with the correct mean spike count.
mean_rate_param = n_spks / n_obs;
hp_g_l_m_ll = sum(y_train' * log(mean_rate_param)) ...
              - (mean_rate_param * n_obs) - sum(log(factorial(y_train)));

% Print log-likelihood values for the different models.
fprintf('The log-likelihood value for the HP-GLM is %.3f\n', hp_g_l_m_ll);
fprintf('The log-likelihood value for the G-GLM is %.3f\n', g_g_l_m_ll);
fprintf('The log-likelihood value for the P-GLM is %.3f\n', p_g_l_m_ll);
fprintf('The log-likelihood value for the NP-GLM is %.3f\n', np_g_l_m_ll);

% We see here that the log-likelihood values align with the performance of
% our models: the models sorted descendingly by log-likelihood values
% are: NP-GLM, P-GLM, G-GLM, HP-GLM.

% Single-spike information:
% ------------------------ 
% The difference of the loglikelihood and homogeneous-Poisson
% loglikelihood, normalized by the number of spikes, gives us an intuitive
% way to compare log-likelihoods in units of bits / spike.  This is a
% quantity known as the (empirical) single-spike information. [See Brenner
% et al, "Synergy in a Neural Code", Neural Comp 2000]. You can think of
% this as the number of bits we know (number of yes/no questions that we
% can answer) about the times of spikes when we know the spike rate output
% by the model, compared to when we only know the (constant) mean spike
% rate.

SSinfo_expGLM = (ll_p_glm - LL0)/n_spks/log(2);
SSinfo_npGLM = (LL_npGLM - LL0)/n_spks/log(2);
% (if we don't divide by log 2 we get it in nats)

fprintf('\n empirical single-spike information:\n ---------------------- \n');
fprintf('exp-GLM: %.2f bits/sp\n',SSinfo_expGLM);
fprintf(' np-GLM: %.2f bits/sp\n',SSinfo_npGLM);

% Let's plot the rate predictions for the two models 
% --------------------------------------------------
subplot(111);
stem(t_in_stim_bins_1s,sps(stim_bins_1s)); hold on;
plot(t_in_stim_bins_1s,y_p_glm(stim_bins_1s),t_in_stim_bins_1s,ratepred_pGLMnp(stim_bins_1s),'linewidth',2); 
hold off; title('rate predictions');
ylabel('spikes / bin'); xlabel('time (s)');
set(gca,'xlim', t_in_stim_bins_1s([1 end]));
legend('spike count', 'exp-GLM', 'np-GLM');


% Akaike information criterion (AIC) is a method for model comparison that
% uses the maximum likelihood, penalized by the number of parameters.
% (This allows us to compensate for the fact that models with more
% parameters can in general achieve higher log-likelihood. AIC determines
% how big this tradeoff should be in terms of the quantity:
%        AIC = - 2*log-likelihood + 2 * number-of-parameters
% The model with lower AIC is 
% their likelihood (at the ML estimate), penalized by the number of parameters  

AIC_expGLM = -2*ll_p_glm + 2*(1+ntfilt); 
AIC_npGLM = -2*LL_npGLM + 2*(1+ntfilt+n_p_npf);

fprintf('\n AIC comparison:\n ---------------------- \n');
fprintf('exp-GLM: %.1f\n',AIC_expGLM);
fprintf(' np-GLM: %.1f\n',AIC_npGLM);
fprintf('\nAIC diff (exp-np)= %.2f\n',AIC_expGLM-AIC_npGLM);
if AIC_expGLM < AIC_npGLM
    fprintf('AIC supports exponential-nonlinearity!\n');
else
    fprintf('AIC supports nonparametric nonlinearity!\n');
    % (despite its greater number of parameters)
end

% Caveat: technically the AIC should be using the maximum of the likelihood
% for a given model.  Here we actually have an underestimate of the
% log-likelihood for the non-parameteric nonlinearity GLM because
% because we left the filter parameters unchanged from the exponential-GLM.
% So a proper AIC comparison (i.e., if we'd achieved a true ML fit) would
% favor the non-parametric nonlinearity GLM even more!

% Exercise: go back and increase 'nfbins', the number of parameters (bins)
% governing the nonparametric nonlinearity. If you increase it enough, you
% should be able to flip the outcome so exponential nonlinearity wins.

% (Note: in the third tutorial we'll use cross-validation to properly
% evaluate the goodness of the fit of the models, e.g., allowing us to
% decide how many bins of stimulus history or how many bins to use for the
% non-parametric nonlinearity, or how to set regularization
% hyperparameters. The basic idea is to split data into training and test
% sets.  Fit the parameters on the training set, and compare models by
% evaluating log-likelihood on test set.)
%% 8d. Reviewing model performance: KS Test on time-rescaled data

%% 8e. Reviewing model performance: MLRT

%% 9. Simulating the GLM & making a raster plot

% Lastly, let's simulate the response of the GLM to a repeated stimulus and
% make raster plots 

% Get chunk of stimulus to repeat, and get model's predicted spike count
% for this chunk.
stim_rpt = stim(stim_bins_1s);         % stimulus to repeat
n_rpts = 50;                           % number of repeats of stim
f_r = np_g_l_m_y_train(stim_bins_1s);  % firing rate (in spike counts)

% Plot.
% First, plot stimulus and true spike counts.
clf
subplot(611);
plot(t_in_stim_bins_1s, stim_rpt, 'linewidth', 2);
axis tight;
title('raw stimulus (full field flicker)');
ylabel('stim intensity'); 
set(gca, 'xticklabel', {});
subplot(612);
% Get and plot the spike counts that happen within `bins`.
stem(t_in_stim_bins_1s, spk_ts_hist(stim_bins_1s), 'linewidth', 2);
set(gca,'xlim', t_in_stim_bins_1s([1 end]));
title('true spike counts');
ylabel('spike count');
set(gca,'xticklabel', {});
% Simulate spikes per bin by outputting a value from a Poisson process with
% rate parameter equal to `f_r` in that bin.
spk_cnts = poissrnd(repmat(f_r', n_rpts, 1));
subplot(6, 1, 3:6);
imagesc(t_in_stim_bins_1s, 1 : n_rpts, spk_cnts);
ylabel('repeat #');
xlabel('time (s)');
title('simulated GLM spike trains');
h_cb = colorbar;
h_cb.Location = 'southoutside';
h_cb.Label.String = 'spike count';

%% 10: Redo using finer time bins, so we report a binary response variable

dt_2 = 0.0001;  % new bin length (.1 ms)
up_s_x = (stim_ts(2) - stim_ts(1)) / dt_2;  % upsample factor for new `dt`
t_in_stim_bins_1s_2 = 0 : dt : 1;

% Compute the fine-time-bin firing rate (which must be scaled down by bin
% width)
f_r_2 = interp1(t_in_stim_bins_1s, f_r, t_in_stim_bins_1s_2, ...
                'nearest', 'extrap') ./ up_s_x;

% now draw fine-timescale spike train
spk_cnts_2 = poissrnd(repmat(f_r_2, n_rpts, 1));

% Re-make plot.
subplot(6, 1, 2)
% Now plot spike raster instead of spike counts.
spk_ts_in_1s = ...
    spk_ts_cell((spk_ts_cell >= t_in_stim_bins_1s(1)) ...
                 & (spk_ts_cell < t_in_stim_bins_1s(end)));
plot(spk_ts_in_1s, 1, 'bo');
subplot(6,1,3:6);
imagesc(t_in_stim_bins_1s_2, 1 : n_rpts, spk_cnts_2);
ylabel('repeat #');
xlabel('time (s)');
title('simulated GLM spike trains');
h_cb = colorbar;
h_cb.Location = 'southoutside';
h_cb.Label.String = 'spike count';

%% Suggested Exercises (advanced)
% -------------------------------
%
% 1) Go back and try it out for the other three neurons!
% (Go to block 1 and change the variable 'cellnum' to 1, 2, or 3.)
% 
% 2) Write your own code to do maximum likelihood estimation of the filter
% and the nonlinearity.  Your function should take in the parameters for
% the filter and the nonlinearity, and compute the Poisson log-likelihood
% function, the log Probability of the spike responses given the stimuli
% and the parameters.  A nice way to parametrize the nonlinearity is with a
% linear combination of basis functions, e.g.
%      f(x) = sum_i  w_i * f_i(x)
% where f_i(x) is the i'th basis function and w_i is the weight on that
% basis function.  You can choose the f_i to be Gaussian bumps or sigmoids,
% i.e. f_i(x) = 1./(1+exp(-x - c_i)) where c_i is the shift for the i'th
% basis function.
%
% Another alternative (that will prevent negative firing rates) is to
% parameterize the log-firing rate with a linear combination of basis
% functions, e.g.
%  log(f(x)) = sum_i  w_i * f_i(x)
%        meaning that
%  f(x) = exp(sum_i  w_i * f_i(x))
%  Now your weights can be negative or positive without fear of generating
%  negative values (which will cause your negative log-likelihood function
%  to give nans or -infs.  
%  
%  Write a function that takes in k and weight vector w and computes the
%  Poisson log-likelihood.  Hand that function off to fminunc and compare
%  the accuracy of the fits you get to the model with fixed exponential
%  nonlinearity.
