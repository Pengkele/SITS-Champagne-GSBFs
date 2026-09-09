function [source_est, result] = sits_champagne_gsbfs( ...
    leadfield, seeg_pos, source_pos, meg_grad, meg_data, seeg_data, opts)
%SITS_CHAMPAGNE_GSBFS SEEG-informed, two-stage Champagne with volume GSBFs.
% [source_est,result] = sits_champagne_gsbfs(leadfield,seeg_pos, ...
%     source_pos,meg_grad,meg_data,seeg_data,opts)
%
% 单文件实现；辅助函数均在本文件末尾。需要 MATLAB R2016b 或更新版本，
% 仅使用 MATLAB 基础功能，不需要 FieldTrip 或 Statistics Toolbox。
% 每次调用处理一对候选 IED 模式；不同模式请分别调用，不能混在同一平均中。
%
% INPUTS
% leadfield  M-by-N fixed-orientation matrix, M-by-(3*N) Cartesian matrix
%            with columns [x1 y1 z1 x2 y2 z2 ...], M-by-3-by-N array,
%            or N-element cell array of M-by-1 / M-by-3 matrices.
%            Also accepts a scalar FieldTrip-style struct with .leadfield
%            and optional .label. Include ONLY valid modeled source nodes;
%            remove outside-grid/empty cells and matching source_pos rows.
% seeg_pos   E-by-3 SEEG contact coordinates, in the SAME coordinate system
%            and units as source_pos. Bipolar data require corresponding
%            channel positions to have been defined by the caller.
% source_pos N-by-3 modeled source coordinates, in leadfield source order.
% meg_grad   FieldTrip-style sensor struct, or []. Its .label can specify
%            meg_data channel order. Geometry is metadata: the supplied
%            leadfield already contains the forward model; no head model
%            is inferred or recomputed from meg_grad.
% meg_data   M-by-Tm or M-by-Tm-by-Rm: one averaged IED or aligned trials.
% seeg_data  E-by-Ts or E-by-Ts-by-Rs: one averaged IED or aligned trials.
%            Both modalities must be preprocessed, artifact-rejected and
%            independently aligned to the SAME IED pattern at time zero.
%            Sampling rates and trial counts may differ. Trials are never
%            paired across modalities. Data must be real and finite.
%            MEG data and leadfield rows MUST use the same physical units
%            and the same projections/reference/channel preprocessing.
% opts       Scalar struct; required fields:
%   .meg_time           1-by-Tm time vector in seconds, strictly increasing.
%   .seeg_time          1-by-Ts time vector in seconds, strictly increasing.
%   .scales             Positive Gaussian standard deviations, IN mm.
%   .adjacency          N-by-N symmetric connectivity, preferably sparse;
%                       nonzero entries denote allowed edges. Edge lengths
%                       are computed from source_pos in mm. Alternatively,
%   .grid_spacing       Scalar or [dx dy dz] IN mm for an axis-aligned
%                       regular volume lattice; missing voxels stay absent.
%                       Required only when adjacency is not supplied.
%
% OPTIONAL OPTIONS (defaults in parentheses)
%   .position_unit      'mm', 'cm', or 'm' ('mm'), for BOTH coordinate arrays.
%   .connectivity       6, 18, or 26 (6), for regular-grid graph construction.
%   .baseline_window    [-0.5 -0.2] seconds; used in each modality separately.
%   .peak_window        [0 0] seconds; a fixed SEEG window. A zero-width
%                       window selects the sample nearest the given time.
%   .match_window       [] = common recorded interval; or [start end] s.
%   .inverse_window     [] = all MEG samples; or [start end] s.
%   .gfp_threshold      1/sqrt(2); positive Pearson correlation, not abs(r).
%   .alpha              0.001; TWO-sided Bonferroni-adjusted p < alpha.
%   .seeg_t             E-by-1 precomputed t statistics (default []).
%   .seeg_p             E-by-1 corresponding RAW two-sided p values ([]).
%                       Both are REQUIRED for already-averaged SEEG input.
%                       Convention: positive t means peak > baseline in
%                       amplitude; negate baseline-minus-peak t values.
%                       They override internal trial statistics if given.
%   .secondary_radius   8 mm; geodesic radius around snapped primary seeds.
%   .max_seed_distance  8 mm; reject contacts farther from the source grid.
%   .noise_cov          M-by-M covariance in original MEG data units ([] =
%                       estimate from the baseline of the averaged MEG).
%   .noise_rank_tol     1e-10; retain eigenvalues > tol * largest eigenvalue.
%   .source_orientation N-by-3 unit directions ([]). If supplied, project
%                       three-component leadfields before constructing GSBFs.
%                       Otherwise use three Cartesian components with one
%                       shared ARD variance per GSBF (rotation invariant).
%   .meg_labels         Channel labels in meg_data row order ([]).
%   .leadfield_labels   Leadfield row labels ([], or struct .label).
%                       Labelled leadfields are reordered to meg_data order.
%                       With unlabelled matrices, row order is caller-owned.
%   .max_iter           500 Champagne-NL updates.
%   .min_iter           5 updates before accepting convergence.
%   .tol                1e-6 relative evidence AND variance-change tolerance.
%   .noise_floor        1e-10, relative to normalized fitting-data power.
%   .verbose            false; print convergence information if true.
%
% OUTPUTS
% source_est N-by-Tfit for fixed orientation; N-by-3-by-Tfit otherwise.
%            Signed source estimates in the units implied by data/leadfield.
% result     Includes times, GFP match, electrode statistics, graph and seed
%            metadata, normalized GSBF matrix, posterior coefficients,
%            source magnitudes, peak map, noise estimates and convergence.
%            source_map is the magnitude at the MEG sample nearest t=0;
%            it is continuous and has NO spatial threshold or smoothing.
%
% EXAMPLE (scale choices below are illustrative, not reported paper values)
% opts.meg_time = (-500:500)/1000;
% opts.seeg_time = (-1024:1024)/2048;
% opts.scales = [8 16 24];
% opts.grid_spacing = 8;
% [S,info] = sits_champagne_gsbfs(L,seeg_pos,source_pos,grad, ...
%                              meg_epochs,seeg_epochs,opts);
% % For averaged SEEG, also supply opts.seeg_t and opts.seeg_p from the
% % original trial analysis, with the sign convention documented above.
%
% METHOD AND MANUSCRIPT DETAILS
% 1. Baseline-correct trials and average each modality separately. Compute
%    GFP = sqrt(mean(B.^2,1)), as Eqs. (15)-(16); do not spatially demean.
%    Interpolate GFP only (not inverse data) to the coarser common time grid.
%    Reject non-concordant pairs using Eq. (17).
% 2. Select SEEG contacts with two-sided tests and Bonferroni correction.
%    The manuscript does not identify the test's sampling unit. Here the
%    independent unit is the SEEG trial: compare mean ABSOLUTE amplitude in
%    the fixed peak window with mean absolute baseline amplitude within
%    each baseline-corrected trial, then test paired differences. Retain
%    positive effects only. This explicit implementation choice handles
%    either spike polarity without treating correlated time samples as
%    independent trials. At least two trials or external t/p are needed.
% 3. Map retained contacts to nearest volume nodes; tied/duplicate anchors
%    keep the largest contact rank weight. Secondary centers are within the
%    specified graph radius. Distances are shortest weighted graph paths,
%    NOT straight-line distances between all source nodes (Eq. 18).
%    Rank weight = ascending average rank / number of retained contacts.
%    The manuscript leaves n's shell definition unspecified: here n is
%    one plus unweighted graph-hop distance from its parent primary node.
% 4. Eqs. (19)-(22) are implemented LITERALLY:
%      phi = (rank/n^2) * exp(-distance.^2/(2*sigma^2)); W = phi/norm(phi).
%    IMPORTANT: each positive scalar rank/n^2 CANCELS in this normalization.
%    Thus the final basis does not encode the claimed scalar attenuation.
%    We report the raw weights, but do not silently move them after the
%    normalization or introduce a different prior. This inconsistency in
%    the manuscript requires a methodological decision if weights are
%    intended to influence the inverse solution.
% 5. SEEG seeds replace the first MEG-only sparse-screening stage. Fit
%    Champagne-NL directly in F=L*W, with simultaneous convex-bound source
%    and diagonal noise updates, Eqs. (6)-(8),(12)-(14). Eq. (13) prints an
%    inverse on its covariance definition; use C=F*Gamma*F'+Lambda.
%    In the three-component extension Gamma has gamma(k)*eye(3) blocks.
%    Whitening is applied to BOTH data and leadfield. All inverse operations
%    use Cholesky solves; no dense N-by-N source covariance is constructed.
%
% The manuscript does not fix numerical scales, graph connectivity, rank
% mapping, test sampling unit, or convergence settings. These are exposed
% above; defaults are implementation choices where not specified. The
% reported 8 mm sensitivity radius assumes coordinates have been converted
% correctly. No random initialization, file I/O or hidden workspace state.
%
% References: supplied SITS-Champagne-GSBFs manuscript, Eqs. (1)-(22).
% Li et al., IEEE TMI (2025), doi:10.1109/TMI.2024.3462415.
% Cai et al., NeuroImage (2021), doi:10.1016/j.neuroimage.2020.117411.
% Champagne-NL source: https://pmc.ncbi.nlm.nih.gov/articles/PMC8451305/

if nargin ~= 7
    error('SITS:Inputs', 'Provide six data inputs followed by an opts struct.');
end
opts = local_options(opts);
validateattributes(source_pos, {'numeric'}, ...
    {'real','finite','nonempty','2d','ncols',3}, mfilename, 'source_pos');
validateattributes(seeg_pos, {'numeric'}, ...
    {'real','finite','nonempty','2d','ncols',3}, mfilename, 'seeg_pos');
local_data_check(meg_data, 'meg_data');
local_data_check(seeg_data, 'seeg_data');
M = size(meg_data,1);
E = size(seeg_data,1);
N = size(source_pos,1);
if size(seeg_pos,1) ~= E
    error('SITS:SEEGOrder', 'seeg_pos must have one row per seeg_data channel.');
end
unit_factor = local_unit_factor(opts.position_unit);
pos_mm = double(source_pos) * unit_factor;
contacts_mm = double(seeg_pos) * unit_factor;
if size(unique(pos_mm,'rows'),1) ~= N
    error('SITS:DuplicateNodes', 'source_pos contains duplicate source nodes.');
end
[L, nori, labels] = local_leadfield(leadfield, meg_grad, M, N, opts);
tm = local_time(opts.meg_time, size(meg_data,2), 'meg_time');
ts = local_time(opts.seeg_time, size(seeg_data,2), 'seeg_time');
bm = local_window(tm, opts.baseline_window, 'MEG baseline');
bs = local_window(ts, opts.baseline_window, 'SEEG baseline');
if numel(bm) < 2 || numel(bs) < 2
    error('SITS:Baseline', 'Each baseline must contain at least two samples.');
end
pk = local_window(ts, opts.peak_window, 'SEEG peak');
if ~isempty(intersect(pk,bs))
    error('SITS:OverlappingWindows', 'Peak and baseline windows must not overlap.');
end
meg_trials = double(meg_data);
meg_trials = bsxfun(@minus, meg_trials, mean(meg_trials(:,bm,:),2));
seeg_trials = double(seeg_data);
seeg_trials = bsxfun(@minus, seeg_trials, mean(seeg_trials(:,bs,:),2));
Bmeg = mean(meg_trials,3);
Bseeg = mean(seeg_trials,3);
gfp_meg = sqrt(mean(Bmeg.^2,1));
gfp_seeg = sqrt(mean(Bseeg.^2,1));
[gfp_r, match_time, matched_gfp] = local_gfp_match( ...
    tm, ts, gfp_meg, gfp_seeg, opts.match_window);
if gfp_r < opts.gfp_threshold
    error('SITS:UnmatchedPattern', ...
        'GFP correlation %.6g is below %.6g. No SEEG-informed fit performed.', ...
        gfp_r, opts.gfp_threshold);
end

% Contact inference precedes any source-space fitting.
stats = local_contact_statistics(seeg_trials, pk, bs, opts);
retained = find(stats.t > 0 & stats.p_bonferroni < opts.alpha);
if isempty(retained)
    error('SITS:NoSignificantContacts', ...
        'No positive SEEG amplitude effect survives Bonferroni p < %.6g.', opts.alpha);
end
stats.retained = retained;
stats.rank_weight = zeros(E,1);
stats.rank_weight(retained) = local_average_ranks(stats.t(retained)) / numel(retained);

% Anatomically restricted source graph and multiscale Gaussian dictionary.
[G, graph_description] = local_source_graph(pos_mm, opts);
[Phi, basis_info] = local_gsbfs(G, pos_mm, contacts_mm, ...
    retained, stats.rank_weight, opts);
K = size(Phi,2);
if nori == 1
    F = L * Phi;
else
    % Interleaved Cartesian components, shared spatial kernels.
    F = L * kron(sparse(Phi), speye(nori));
end
F = full(F);

% Noise covariance of the AVERAGED baseline, as in the manuscript.
if isempty(opts.noise_cov)
    baseline_data = Bmeg(:,bm);
    baseline_data = bsxfun(@minus, baseline_data, mean(baseline_data,2));
    Cnoise = baseline_data * baseline_data' / (numel(bm)-1);
else
    Cnoise = opts.noise_cov;
    validateattributes(Cnoise, {'numeric'}, ...
        {'real','finite','size',[M M]}, mfilename, 'opts.noise_cov');
    Cnoise = full(double(Cnoise));
end
[whitener, dewhitener, noise_eigenvalues] = local_whitening(Cnoise, opts.noise_rank_tol);
if isempty(opts.inverse_window)
    fit_index = 1:numel(tm);
else
    fit_index = local_window(tm, opts.inverse_window, 'MEG inverse');
end
if numel(fit_index) < 2
    error('SITS:FitWindow', 'The inverse window must contain at least two samples.');
end
if tm(fit_index(1)) > 0 || tm(fit_index(end)) < 0
    error('SITS:FitWindow', 'The inverse window must include the IED peak at 0 s.');
end
Ywhite = whitener * Bmeg(:,fit_index);
Fwhite = whitener * F;

% GLOBAL scaling only; per-column leadfield normalization would change ARD.
y_scale = norm(Ywhite,'fro') / sqrt(numel(Ywhite));
f_scale = norm(Fwhite,'fro') / sqrt(size(Fwhite,2));
if ~(isfinite(y_scale) && y_scale > 0 && isfinite(f_scale) && f_scale > 0)
    error('SITS:DegenerateModel', 'Whitened data or GSBF leadfield has zero/invalid power.');
end
Yn = Ywhite / y_scale;
Fn = Fwhite / f_scale;
% Whitened baseline covariance is identity before fitting-data rescaling.
lambda0 = ones(size(Yn,1),1) / y_scale^2;
if any(~isfinite(lambda0))
    error('SITS:DynamicRange', 'Noise/data scaling exceeds floating-point range.');
end
[Zscaled, gamma_scaled, lambda_scaled, fit] = local_champagne(Fn, Yn, nori, lambda0, opts);
Z = Zscaled * (y_scale/f_scale);
gamma = gamma_scaled * (y_scale/f_scale)^2;
lambda_white = lambda_scaled * y_scale^2;

if nori == 1
    source_est = Phi * Z;
    magnitude = abs(source_est);
else
    source_est = zeros(N,nori,numel(fit_index));
    for component = 1:nori
        X = Phi * Z(component:nori:end,:);
        source_est(:,component,:) = reshape(X,N,1,numel(fit_index));
    end
    magnitude = reshape(sqrt(sum(source_est.^2,2)),N,numel(fit_index));
end
if any(~isfinite(source_est(:)))
    error('SITS:DynamicRange', 'Source reconstruction exceeds floating-point range.');
end
[~, zero_index] = min(abs(tm(fit_index)));
predicted_meg = F * Z;

result = struct();
result.method = 'SITS-Champagne-GSBFs';
result.time = tm(fit_index);
result.meg_labels = labels;
result.n_orientations = nori;
result.source_pos = source_pos;
result.position_unit = opts.position_unit;
result.source_magnitude = magnitude;
result.source_map = magnitude(:,zero_index);
result.source_map_time = result.time(zero_index);
result.source_rms = sqrt(mean(magnitude.^2,2));
result.gfp = struct('meg',gfp_meg,'seeg',gfp_seeg, ...
    'meg_time',tm,'seeg_time',ts,'common_time',match_time, ...
    'common_profiles',matched_gfp,'correlation',gfp_r, ...
    'threshold',opts.gfp_threshold);
result.seeg_statistics = stats;
result.basis = basis_info;
result.basis.Phi = Phi;
result.basis.n_columns = K;
result.basis.scalar_weights_cancel_after_l2 = true;
result.graph = struct('description',graph_description,'n_nodes',numnodes(G), ...
    'n_edges',numedges(G),'edge_distance_unit','mm');
result.coefficients = Z; % K rows, or 3*K interleaved rows.
result.gamma = gamma;   % One source variance per GSBF; shared across orientations.
result.noise_variance_whitened = lambda_white;
% If rank was reduced, this covariance describes only the retained subspace.
result.noise_cov_retained_sensor_space = ...
    bsxfun(@times,dewhitener,lambda_white') * dewhitener';
result.whitener = whitener;
result.noise_rank = size(whitener,1);
result.noise_eigenvalues_retained = noise_eigenvalues;
result.predicted_meg = predicted_meg; % Baseline-corrected original sensor units.
result.residual_meg = Bmeg(:,fit_index) - predicted_meg;
result.fit = fit;
result.fit.coefficient_scale = y_scale/f_scale;
result.fit.data_scale = y_scale;
result.fit.leadfield_scale = f_scale;
result.fit.cost_definition = 'logdet(C) + trace(C\(Y*Y''/T)), in normalized whitened units';
result.n_trials = [size(meg_data,3),size(seeg_data,3)];
result.options = opts;
end

function opts = local_options(opts)
if ~isstruct(opts) || ~isscalar(opts)
    error('SITS:Options', 'opts must be a scalar struct.');
end
defaults = struct('meg_time',[],'seeg_time',[],'scales',[], ...
    'adjacency',[],'grid_spacing',[],'position_unit','mm','connectivity',6, ...
    'baseline_window',[-0.5 -0.2],'peak_window',[0 0], ...
    'match_window',[],'inverse_window',[],'gfp_threshold',1/sqrt(2), ...
    'alpha',0.001,'seeg_t',[],'seeg_p',[],'secondary_radius',8, ...
    'max_seed_distance',8,'noise_cov',[],'noise_rank_tol',1e-10, ...
    'source_orientation',[],'meg_labels',[],'leadfield_labels',[], ...
    'max_iter',500,'min_iter',5,'tol',1e-6,'noise_floor',1e-10,'verbose',false);
unknown = setdiff(fieldnames(opts),fieldnames(defaults));
if ~isempty(unknown)
    error('SITS:Options', 'Unknown option: %s.', unknown{1});
end
names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(opts,names{k})
        opts.(names{k}) = defaults.(names{k});
    end
end
if isempty(opts.meg_time) || isempty(opts.seeg_time) || isempty(opts.scales)
    error('SITS:Options', 'opts.meg_time, opts.seeg_time and opts.scales are required.');
end
if isempty(opts.adjacency) && isempty(opts.grid_spacing)
    error('SITS:Options', 'Supply opts.adjacency or opts.grid_spacing to define geodesic paths.');
end
validateattributes(opts.scales,{'numeric'},{'real','finite','vector','positive','nonempty'});
opts.scales = unique(double(opts.scales(:)'),'stable');
validateattributes(opts.gfp_threshold,{'numeric'},{'real','finite','scalar','>=',0,'<=',1});
validateattributes(opts.alpha,{'numeric'},{'real','finite','scalar','>',0,'<',1});
validateattributes(opts.secondary_radius,{'numeric'},{'real','finite','scalar','nonnegative'});
validateattributes(opts.max_seed_distance,{'numeric'},{'real','finite','scalar','nonnegative'});
validateattributes(opts.noise_rank_tol,{'numeric'},{'real','finite','scalar','>',0,'<',1});
validateattributes(opts.max_iter,{'numeric'},{'real','finite','scalar','integer','positive'});
validateattributes(opts.min_iter,{'numeric'},{'real','finite','scalar','integer','positive','<=',opts.max_iter});
validateattributes(opts.tol,{'numeric'},{'real','finite','scalar','>',0,'<',1});
validateattributes(opts.noise_floor,{'numeric'},{'real','finite','scalar','>',0,'<',1});
validateattributes(opts.verbose,{'logical','numeric'},{'scalar','binary'});
validateattributes(opts.connectivity,{'numeric'},{'real','finite','scalar'});
if ~ismember(opts.connectivity,[6 18 26])
    error('SITS:Options', 'opts.connectivity must be 6, 18 or 26.');
end
opts.verbose = logical(opts.verbose);
end

function local_data_check(X, name)
validateattributes(X,{'numeric'},{'real','finite','nonempty'},mfilename,name);
if ndims(X) > 3 || size(X,2) < 3 || issparse(X)
    error('SITS:DataShape', '%s must be a full channels-by-time[-by-trials] array.',name);
end
end

function factor = local_unit_factor(unit)
if ~(ischar(unit) || (isstring(unit) && isscalar(unit)))
    error('SITS:Units','position_unit must be mm, cm or m.');
end
switch lower(char(unit))
    case 'mm', factor = 1;
    case 'cm', factor = 10;
    case 'm', factor = 1000;
    otherwise, error('SITS:Units','position_unit must be mm, cm or m.');
end
end

function t = local_time(t, T, name)
validateattributes(t,{'numeric'},{'real','finite','vector','numel',T},mfilename,name);
t = double(t(:)');
if any(diff(t) <= 0)
    error('SITS:Time', '%s must increase strictly.',name);
end
end

function ind = local_window(t, interval, name)
validateattributes(interval,{'numeric'},{'real','finite','vector','numel',2});
interval = double(interval(:)');
% A relative numerical allowance only; do not silently shorten windows.
slack = 32*eps(max(1,max(abs(t))));
if interval(2) < interval(1) || interval(1) < t(1)-slack || interval(2) > t(end)+slack
    error('SITS:Window', '%s window is reversed or outside its time axis.',name);
end
if interval(1) == interval(2)
    [~,ind] = min(abs(t-interval(1)));
else
    ind = find(t >= interval(1)-slack & t <= interval(2)+slack);
end
if isempty(ind)
    error('SITS:Window', '%s window contains no samples.',name);
end
end

function labels = local_labels(labels, name)
if isstring(labels)
    labels = cellstr(labels(:));
elseif ischar(labels)
    labels = cellstr(labels);
end
if ~iscell(labels) || ~all(cellfun(@ischar,labels(:)))
    error('SITS:Labels', '%s must be a cell array of character vectors or strings.',name);
end
labels = labels(:);
if numel(unique(labels)) ~= numel(labels)
    error('SITS:Labels', '%s contains duplicate labels.',name);
end
end

function [L, nori, target] = local_leadfield(input, grad, M, N, opts)
lf_labels = opts.leadfield_labels;
if isstruct(input)
    if ~isscalar(input) || ~isfield(input,'leadfield')
        error('SITS:Leadfield','Leadfield structs require a .leadfield field.');
    end
    if isempty(lf_labels) && isfield(input,'label')
        lf_labels = input.label;
    end
    input = input.leadfield;
end
if iscell(input)
    if numel(input) ~= N || isempty(input{1})
        error('SITS:Leadfield','Provide one nonempty leadfield cell per source node.');
    end
    nori = size(input{1},2);
    nrow = size(input{1},1);
    if ~ismember(nori,[1 3])
        error('SITS:Leadfield','Each leadfield cell must have 1 or 3 columns.');
    end
    for k = 1:N
        validateattributes(input{k},{'numeric'}, ...
            {'real','finite','size',[nrow nori]},mfilename,'leadfield cell');
    end
    L = double(cat(2,input{:}));
else
    validateattributes(input,{'numeric'},{'real','finite','nonempty'});
    if ndims(input) == 3
        if size(input,2) ~= 3 || size(input,3) ~= N
            error('SITS:Leadfield','Three-dimensional leadfield must be M-by-3-by-N.');
        end
        nori = 3;
        L = reshape(double(input),size(input,1),3*N);
    elseif ismatrix(input) && ismember(size(input,2),[N 3*N])
        nori = size(input,2)/N;
        L = double(input);
    else
        error('SITS:Leadfield','Leadfield must have N or 3*N interleaved columns.');
    end
end
if ~isempty(grad) && (~isstruct(grad) || ~isscalar(grad))
    error('SITS:Grad','meg_grad must be a scalar sensor struct or [].');
end
target = opts.meg_labels;
if isempty(target) && isstruct(grad) && isfield(grad,'label')
    target = grad.label;
end
if ~isempty(lf_labels)
    lf_labels = local_labels(lf_labels,'leadfield labels');
    if numel(lf_labels) ~= size(L,1)
        error('SITS:Labels','Leadfield label count differs from its row count.');
    end
end
if isempty(target) && ~isempty(lf_labels)
    target = lf_labels;
end
if ~isempty(target)
    target = local_labels(target,'MEG labels');
    if numel(target) ~= M
        error('SITS:Labels','Provide opts.meg_labels in the exact meg_data row order.');
    end
    if ~isempty(lf_labels)
        [found,order] = ismember(target,lf_labels);
        if ~all(found)
            error('SITS:Labels','A MEG data channel is absent from the leadfield.');
        end
        L = L(order,:);
    end
end
if size(L,1) ~= M
    error('SITS:Leadfield','Leadfield rows must match meg_data channels after label alignment.');
end
if ~isempty(opts.source_orientation)
    if nori ~= 3
        error('SITS:Orientation','source_orientation requires a three-component leadfield.');
    end
    ori = opts.source_orientation;
    validateattributes(ori,{'numeric'},{'real','finite','size',[N 3]});
    ori = double(ori);
    lengths = sqrt(sum(ori.^2,2));
    if any(lengths == 0)
        error('SITS:Orientation','Source directions must be nonzero.');
    end
    ori = bsxfun(@rdivide,ori,lengths);
    Lfixed = zeros(M,N);
    for c = 1:3
        Lfixed = Lfixed + bsxfun(@times,L(:,c:3:end),ori(:,c)');
    end
    L = Lfixed;
    nori = 1;
end
L = full(L);
end

function [r, t, profiles] = local_gfp_match(tm, ts, gm, gs, interval)
lo = max(tm(1),ts(1));
hi = min(tm(end),ts(end));
if ~isempty(interval)
    local_window(tm,interval,'GFP matching (MEG)');
    local_window(ts,interval,'GFP matching (SEEG)');
    lo = interval(1);
    hi = interval(2);
end
step = max(median(diff(tm)),median(diff(ts)));
t = lo:step:hi;
if numel(t) < 3
    error('SITS:GFP','At least three common time samples are required for GFP matching.');
end
profiles = [interp1(tm,gm,t,'linear');interp1(ts,gs,t,'linear')];
if any(~isfinite(profiles(:)))
    error('SITS:GFP','GFP interpolation has invalid values; check times and data scaling.');
end
a = profiles(1,:) - mean(profiles(1,:));
b = profiles(2,:) - mean(profiles(2,:));
na = norm(a);
nb = norm(b);
if na == 0 || nb == 0 || ~isfinite(na) || ~isfinite(nb)
    error('SITS:GFP','GFP correlation is undefined for constant/invalid profiles.');
end
r = max(-1,min(1,(a/na)*(b/nb)'));
end

function stats = local_contact_statistics(X, peak, baseline, opts)
E = size(X,1);
R = size(X,3);
if xor(isempty(opts.seeg_t),isempty(opts.seeg_p))
    error('SITS:Statistics','Supply both opts.seeg_t and opts.seeg_p, or neither.');
end
if ~isempty(opts.seeg_t)
    validateattributes(opts.seeg_t,{'numeric'},{'real','vector','numel',E});
    validateattributes(opts.seeg_p,{'numeric'}, ...
        {'real','finite','vector','numel',E,'>=',0,'<=',1});
    t = double(opts.seeg_t(:));
    p = double(opts.seeg_p(:));
    if any(isnan(t))
        error('SITS:Statistics','External t statistics may contain Inf but not NaN.');
    end
    effect = nan(E,1);
    df = NaN;
    method = 'External two-sided t statistics; positive t = peak amplitude > baseline';
else
    if R < 2
        error('SITS:Statistics', ['Averaged SEEG cannot supply trial variance. Provide ' ...
            'SEEG trials or opts.seeg_t and RAW two-sided opts.seeg_p.']);
    end
    peak_amplitude = reshape(mean(abs(X(:,peak,:)),2),E,R);
    baseline_amplitude = reshape(mean(abs(X(:,baseline,:)),2),E,R);
    differences = peak_amplitude-baseline_amplitude;
    effect = mean(differences,2);
    standard_error = std(differences,0,2)/sqrt(R);
    t = zeros(E,1);
    good = standard_error > 0;
    t(good) = effect(good)./standard_error(good);
    degenerate = ~good & effect ~= 0;
    t(degenerate) = sign(effect(degenerate))*Inf;
    df = R-1;
    % Exact two-sided Student-t tail from the regularized incomplete beta.
    p = betainc(df./(df+t.^2),df/2,0.5);
    method = 'Two-sided paired trial test of mean absolute peak-minus-baseline amplitude';
end
stats = struct('t',t,'t_baseline_minus_peak',-t,'p_raw',p, ...
    'p_bonferroni',min(1,E*p),'amplitude_effect',effect,'degrees_of_freedom',df, ...
    'n_tested_contacts',E,'alpha',opts.alpha,'method',method);
end

function ranks = local_average_ranks(values)
[sorted, order] = sort(values(:),'ascend');
ranks = zeros(numel(values),1);
i = 1;
while i <= numel(sorted)
    j = i;
    while j < numel(sorted) && sorted(j+1) == sorted(i)
        j = j+1;
    end
    ranks(order(i:j)) = (i+j)/2;
    i = j+1;
end
end

function [G, description] = local_source_graph(pos, opts)
N = size(pos,1);
if ~isempty(opts.adjacency)
    A = opts.adjacency;
    validateattributes(A,{'numeric','logical'},{'real','finite','size',[N N]});
    if any(nonzeros(A) < 0)
        error('SITS:Graph','Adjacency entries must be nonnegative.');
    end
    A = spones(sparse(A));
    if nnz(A-A') ~= 0 || any(diag(A))
        error('SITS:Graph','Adjacency must be symmetric with a zero diagonal.');
    end
    [i,j] = find(triu(A,1));
    description = 'Caller-supplied source topology; physical edge lengths';
else
    spacing = opts.grid_spacing;
    validateattributes(spacing,{'numeric'}, ...
        {'real','finite','vector','nonempty','positive'});
    if isscalar(spacing)
        spacing = repmat(double(spacing),1,3);
    elseif numel(spacing) == 3
        spacing = double(spacing(:)');
    else
        error('SITS:Grid','grid_spacing must be scalar or [dx dy dz], in mm.');
    end
    q = bsxfun(@rdivide,bsxfun(@minus,pos,min(pos,[],1)),spacing);
    grid = round(q);
    if any(abs(q(:)-grid(:)) > 1e-5) || size(unique(grid,'rows'),1) ~= N
        error('SITS:Grid', ['source_pos is not an axis-aligned lattice at grid_spacing. ' ...
            'Supply opts.adjacency for irregular or rotated source spaces.']);
    end
    [dx,dy,dz] = ndgrid(-1:1,-1:1,-1:1);
    offsets = [dx(:),dy(:),dz(:)];
    taxicab = sum(abs(offsets),2);
    if opts.connectivity == 6
        offsets = offsets(taxicab == 1,:);
    elseif opts.connectivity == 18
        offsets = offsets(taxicab >= 1 & taxicab <= 2,:);
    else
        offsets = offsets(taxicab >= 1,:);
    end
    % Keep one direction per undirected edge.
    half = offsets(:,1)>0 | (offsets(:,1)==0 & offsets(:,2)>0) | ...
        (offsets(:,1)==0 & offsets(:,2)==0 & offsets(:,3)>0);
    offsets = offsets(half,:);
    ii = cell(size(offsets,1),1);
    jj = ii;
    for k = 1:size(offsets,1)
        [present, neighbor] = ismember(bsxfun(@plus,grid,offsets(k,:)),grid,'rows');
        ii{k} = find(present);
        jj{k} = neighbor(present);
    end
    i = vertcat(ii{:});
    j = vertcat(jj{:});
    description = sprintf('%d-neighbor regular volume graph; physical edge lengths',opts.connectivity);
end
edge_length = sqrt(sum((pos(i,:)-pos(j,:)).^2,2));
G = graph(i,j,edge_length,N);
if N > 1 && numedges(G) == 0
    error('SITS:Graph','No source neighbors found; check graph, grid spacing and coordinate units.');
end
end

function [Phi, info] = local_gsbfs(G, pos, contacts, retained, rank_weight, opts)
N = size(pos,1);
nearest = zeros(numel(retained),1);
snap_distance = zeros(numel(retained),1);
for k = 1:numel(retained)
    d2 = sum(bsxfun(@minus,pos,contacts(retained(k),:)).^2,2);
    [minimum,nearest(k)] = min(d2);
    snap_distance(k) = sqrt(minimum);
end
if any(snap_distance > opts.max_seed_distance)
    bad = find(snap_distance > opts.max_seed_distance,1);
    error('SITS:Registration', ...
        'Retained contact %d is %.6g mm from the grid (limit %.6g mm).', ...
        retained(bad),snap_distance(bad),opts.max_seed_distance);
end
[primary,~,group] = unique(nearest);
primary_rank = accumarray(group,rank_weight(retained),[],@max);
raw_alpha = zeros(N,1);
parent = zeros(N,1);
shell = zeros(N,1);
anchor_rank = zeros(N,1);
for k = 1:numel(primary)
    distance = distances(G,primary(k))';
    hop = distances(G,primary(k),'Method','unweighted')';
    candidates = find(isfinite(distance) & distance <= opts.secondary_radius);
    order = 1+hop(candidates);
    candidate_alpha = primary_rank(k)./(order.^2);
    replace = candidate_alpha > raw_alpha(candidates);
    nodes = candidates(replace);
    raw_alpha(nodes) = candidate_alpha(replace);
    parent(nodes) = primary(k);
    shell(nodes) = order(replace);
    anchor_rank(nodes) = primary_rank(k);
end
% A primary node stays primary even when another primary lies nearby.
raw_alpha(primary) = primary_rank;
parent(primary) = primary;
shell(primary) = 1;
anchor_rank(primary) = primary_rank;
centers = find(raw_alpha > 0);
Q = numel(opts.scales);
Phi = zeros(N,numel(centers)*Q);
column_center = zeros(numel(centers)*Q,1);
column_sigma = column_center;
for k = 1:numel(centers)
    center = centers(k);
    d = distances(G,center)';
    for s = 1:Q
        column = (k-1)*Q+s;
        phi = raw_alpha(center) * exp(-0.5*(d/opts.scales(s)).^2);
        % Disconnected components have d=Inf and therefore zero weight.
        Phi(:,column) = phi/norm(phi,2); % Literal Eq. (21): alpha cancels.
        column_center(column) = center;
        column_sigma(column) = opts.scales(s);
    end
end
info = struct('retained_contacts',retained,'contact_primary_node',nearest, ...
    'contact_snap_distance_mm',snap_distance,'primary_nodes',primary, ...
    'primary_rank_weight',primary_rank,'centers',centers, ...
    'parent_primary_node',parent(centers),'shell_n',shell(centers), ...
    'anchor_rank_weight',anchor_rank(centers),'raw_alpha',raw_alpha(centers), ...
    'column_center_node',column_center,'column_sigma_mm',column_sigma, ...
    'secondary_radius_mm',opts.secondary_radius, ...
    'shell_definition','n = 1 + shortest unweighted hop count from parent');
end

function [W, Winverse, values] = local_whitening(C, rank_tol)
scale = norm(C,'fro');
if ~isfinite(scale) || scale == 0
    error('SITS:Noise','Baseline noise covariance has zero or invalid power.');
end
if norm(C-C','fro') > 1e-10*scale
    error('SITS:Noise','Noise covariance must be symmetric.');
end
C = (C+C')/2;
[U,D] = eig(C);
[values,order] = sort(real(diag(D)),'descend');
U = U(:,order);
if values(1) <= 0 || any(values < -1e-10*values(1))
    error('SITS:Noise','Noise covariance must be positive semidefinite.');
end
keep = values > rank_tol*values(1);
values = values(keep);
U = U(:,keep);
W = bsxfun(@rdivide,U',sqrt(values));
Winverse = bsxfun(@times,U,sqrt(values)');
end

function [Z, gamma, lambda, fit] = local_champagne(F, Y, nori, lambda0, opts)
% One ARD block per spatial basis; no orientation-dependent hyperparameters.
R = size(F,1);
K = size(F,2)/nori;
data_power = norm(Y,'fro')^2/numel(Y);
floor_value = opts.noise_floor*data_power;
lambda = max(lambda0,floor_value);
lead_power = norm(F,'fro')^2/R;
initial_source_power = max(data_power-mean(lambda),0.1*data_power);
gamma = repmat(initial_source_power/lead_power,K,1);
cost_history = nan(opts.max_iter+1,1);
relative_cost = Inf;
relative_variance = Inf;
converged = false;
[Z, sensitivity, inverse_diagonal, cost] = local_posterior(F,Y,gamma,lambda,nori);
cost_history(1) = cost;
for iteration = 1:opts.max_iter
    coefficient_power = sum(reshape(mean(Z.^2,2),nori,K),1)';
    gamma_new = zeros(K,1);
    identifiable = sensitivity > 0;
    gamma_new(identifiable) = sqrt(coefficient_power(identifiable)./sensitivity(identifiable));
    residual = Y-F*Z;
    lambda_new = max(sqrt(mean(residual.^2,2)./inverse_diagonal),floor_value);
    if any(~isfinite(gamma_new)) || any(~isfinite(lambda_new))
        error('SITS:Numerics','Nonfinite Champagne update; check model scaling or noise_floor.');
    end
    % Both updates above use the SAME current posterior/covariance.
    relative_variance = max( ...
        norm(gamma_new-gamma)/max(norm(gamma),realmin), ...
        norm(lambda_new-lambda)/max(norm(lambda),realmin));
    gamma = gamma_new;
    lambda = lambda_new;
    previous_cost = cost;
    [Z,sensitivity,inverse_diagonal,cost] = local_posterior(F,Y,gamma,lambda,nori);
    cost_history(iteration+1) = cost;
    relative_cost = abs(cost-previous_cost)/max(1,abs(previous_cost));
    if opts.verbose && (iteration == 1 || mod(iteration,25) == 0)
        fprintf('SITS: iter %d, cost %.9g, dCost %.3g, dVariance %.3g\n', ...
            iteration,cost,relative_cost,relative_variance);
    end
    if iteration >= opts.min_iter && relative_cost <= opts.tol && relative_variance <= opts.tol
        converged = true;
        break
    end
end
if ~converged
    warning('SITS:MaxIterations', ...
        'Champagne reached %d updates before tolerance; inspect result.fit.',iteration);
end
fit = struct('converged',converged,'iterations',iteration, ...
    'cost_history',cost_history(1:iteration+1),'relative_cost_change',relative_cost, ...
    'relative_variance_change',relative_variance,'noise_floor_normalized',floor_value);
end

function [Z, sensitivity, inverse_diagonal, cost] = local_posterior(F,Y,gamma,lambda,nori)
% nori=1 gives the manuscript's scalar Eqs. (6)-(8) exactly.
expanded_gamma = reshape(repmat(gamma',nori,1),[],1);
C = bsxfun(@times,F,expanded_gamma')*F' + diag(lambda);
C = (C+C')/2;
[R,flag] = chol(C,'lower');
if flag ~= 0
    error('SITS:Covariance', ...
        'Covariance factorization failed. Increase noise_floor or noise_rank_tol.');
end
RY = R\Y;
RF = R\F;
Ri = R\eye(size(R,1));
Z = bsxfun(@times,expanded_gamma,F'*(R'\RY));
sensitivity = sum(reshape(sum(RF.^2,1),nori,[]),1)';
inverse_diagonal = sum(Ri.^2,1)';
cost = 2*sum(log(diag(R))) + sum(RY(:).^2)/size(Y,2);
if ~isfinite(cost)
    error('SITS:Numerics','Nonfinite marginal likelihood; check input dynamic range.');
end
end
