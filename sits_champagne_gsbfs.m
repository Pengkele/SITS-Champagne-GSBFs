function [source_est, result] = sits_champagne_gsbfs( ...
    leadfield, seeg_pos, source_pos, meg_grad, meg_data, seeg_data, opts)
%SITS_CHAMPAGNE_GSBFS SEEG-informed MEG source reconstruction.
%
% leadfield  : MEG channels x sources, with fixed dipole orientations.
% source_pos : sources x 3; seeg_pos: contacts x 3, both in mm.
% meg_data   : channels x time, or channels x time x trials.
% seeg_data  : contacts x time x trials, aligned to the IED peak at t = 0.
% meg_grad   : sensor metadata, saved in result.
%
% All inputs must use matching channel/source order and coordinate frames.
% MEG data and leadfield must use matching sensor units and preprocessing.
% Each call uses one IED pattern; sampling rates and trial counts may differ.
%
% Required opts fields:
%   meg_time, seeg_time : sample times in seconds.
%   scales             : Gaussian standard deviations in mm.
%
% The default source grid is an axis-aligned 8 mm lattice. Set grid_spacing
% for another spacing, or supply a symmetric sparse adjacency matrix.
% Other defaults are listed below. For averaged SEEG, supply seeg_t and
% raw two-sided seeg_p; positive t denotes peak amplitude above baseline.
%
% source_est contains signed source time courses. result.source_map is the
% absolute source amplitude at t = 0, without thresholding or smoothing.
%
% Example:
%   opts.meg_time = (-500:500)/1000;
%   opts.seeg_time = (-1024:1024)/2048;
%   opts.scales = [8 16 24];          % Choose scales for the source grid.
%   opts.grid_spacing = 8;
%   [S, result] = sits_champagne_gsbfs(L, seeg_pos, source_pos, grad, ...
%                                     meg_epochs, seeg_epochs, opts);
%
% Li et al., IEEE TMI, 2025. doi:10.1109/TMI.2024.3462415
% Cai et al., NeuroImage, 2021. doi:10.1016/j.neuroimage.2020.117411


defaults = struct('baseline_window',[-0.5 -0.2], 'grid_spacing',8, ...
    'adjacency',[], 'secondary_radius',8, 'alpha',0.001, ...
    'seeg_t',[], 'seeg_p',[], 'max_iter',500, 'tol',1e-6, ...
    'noise_rank_tol',1e-10, 'noise_floor',1e-10);
names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(opts,names{k})
        opts.(names{k}) = defaults.(names{k});
    end
end

leadfield = double(leadfield);
source_pos = double(source_pos);
seeg_pos = double(seeg_pos);
tm = opts.meg_time(:)';
ts = opts.seeg_time(:)';
nsource = size(source_pos,1);
ncontact = size(seeg_data,1);
ntrial = size(seeg_data,3);

% Baseline correction and averaging.
bm = tm >= opts.baseline_window(1) & tm <= opts.baseline_window(2);
bs = ts >= opts.baseline_window(1) & ts <= opts.baseline_window(2);
meg = mean(double(meg_data),3);
meg = meg - mean(meg(:,bm),2);
seeg = double(seeg_data);
seeg = seeg - mean(seeg(:,bs,:),2);
seeg_avg = mean(seeg,3);

% GFP matching, Eqs. (15)-(17).
gfp_meg = sqrt(mean(meg.^2,1));
gfp_seeg = sqrt(mean(seeg_avg.^2,1));
dt = max(median(diff(tm)),median(diff(ts)));
t = max(tm(1),ts(1)):dt:min(tm(end),ts(end));
gm = interp1(tm,gfp_meg,t);
gs = interp1(ts,gfp_seeg,t);
a = gm - mean(gm);
b = gs - mean(gs);
r = (a/norm(a)) * (b/norm(b))';
if ~isfinite(r) || r < 1/sqrt(2)
    error('SITS:Pattern','MEG and SEEG GFP patterns do not match (r = %.3f).',r);
end

% Paired peak-minus-baseline amplitude differences across SEEG trials.
% The trial is the sampling unit; this detail is not specified in the paper.
if isempty(opts.seeg_t)
    if ntrial < 2
        error('SITS:Trials','Provide SEEG trials or precomputed seeg_t and seeg_p.');
    end
    [~,peak] = min(abs(ts));
    peak_amp = reshape(abs(seeg(:,peak,:)),ncontact,ntrial);
    base_amp = reshape(mean(abs(seeg(:,bs,:)),2),ncontact,ntrial);
    d = peak_amp - base_amp;
    delta = mean(d,2);
    se = std(d,0,2)/sqrt(ntrial);
    tstat = delta./se;
    tstat(se == 0 & delta == 0) = 0;
    df = ntrial - 1;
    p = betainc(df./(df + tstat.^2),df/2,0.5);
else
    tstat = opts.seeg_t(:);
    p = opts.seeg_p(:);
end
p_bonf = min(1,ncontact*p);
contacts = find(tstat > 0 & p_bonf < opts.alpha);
if isempty(contacts)
    error('SITS:Contacts','No SEEG contacts pass the amplitude test.');
end

% Rank weights; tied t values receive their average rank.
[sorted_t,order] = sort(tstat(contacts));
[~,~,group] = unique(sorted_t);
ranks = accumarray(group,(1:numel(contacts))',[],@mean);
contact_weight = zeros(numel(contacts),1);
contact_weight(order) = ranks(group)/numel(contacts);

% Source graph. Edge weights are physical distances.
if isempty(opts.adjacency)
    grid = round((source_pos - min(source_pos,[],1))./opts.grid_spacing);
    edge_i = cell(3,1);
    edge_j = cell(3,1);
    for axis = 1:3
        neighbor = grid;
        neighbor(:,axis) = neighbor(:,axis) + 1;
        [found,index] = ismember(neighbor,grid,'rows');
        edge_i{axis} = find(found);
        edge_j{axis} = index(found);
    end
    ii = vertcat(edge_i{:});
    jj = vertcat(edge_j{:});
else
    [ii,jj] = find(triu(opts.adjacency,1));
end
edge_length = sqrt(sum((source_pos(ii,:) - source_pos(jj,:)).^2,2));
G = graph(ii,jj,edge_length,nsource);

% Map contacts to primary source nodes.
seed = zeros(numel(contacts),1);
for k = 1:numel(contacts)
    distance2 = sum((source_pos - seeg_pos(contacts(k),:)).^2,2);
    [~,seed(k)] = min(distance2);
end
[primary,~,group] = unique(seed);
primary_weight = accumarray(group,contact_weight,[],@max);

% Secondary nodes within the geodesic radius. Use n = 1 + graph hop count.
alpha = zeros(nsource,1);
for k = 1:numel(primary)
    distance = distances(G,primary(k))';
    hops = distances(G,primary(k),'Method','unweighted')';
    nearby = distance <= opts.secondary_radius;
    weight = primary_weight(k)./(1 + hops).^2;
    alpha(nearby) = max(alpha(nearby),weight(nearby));
end
alpha(primary) = primary_weight;
centers = find(alpha > 0);

% Multiscale GSBFs, Eqs. (18)-(22).
scales = opts.scales(:)';
nscale = numel(scales);
W = zeros(nsource,numel(centers)*nscale);
for k = 1:numel(centers)
    distance = distances(G,centers(k))';
    for s = 1:nscale
        column = (k-1)*nscale + s;
        phi = alpha(centers(k))*exp(-0.5*(distance/scales(s)).^2);
        % The scalar alpha cancels in the L2 normalization of Eq. (21).
        W(:,column) = phi/norm(phi);
    end
end
F = leadfield*W;

% Whiten using the covariance of the averaged MEG baseline.
baseline = meg(:,bm);
baseline = baseline - mean(baseline,2);
Cnoise = baseline*baseline'/(sum(bm)-1);
[U,D] = eig((Cnoise + Cnoise')/2);
eigenvalues = diag(D);
keep = eigenvalues > opts.noise_rank_tol*max(eigenvalues);
U = U(:,keep);
eigenvalues = eigenvalues(keep);
P = U'./sqrt(eigenvalues);
Y = P*meg;
Fwhite = P*F;

% Global scaling for the covariance updates.
yscale = norm(Y,'fro')/sqrt(numel(Y));
fscale = norm(Fwhite,'fro')/sqrt(size(Fwhite,2));
Y = Y/yscale;
A = Fwhite/fscale;
nbasis = size(A,2);
nsensor = size(A,1);
lambda = max(1/yscale^2,opts.noise_floor)*ones(nsensor,1);
initial_power = max(1-mean(lambda),0.1);
gamma = initial_power/(norm(A,'fro')^2/nsensor)*ones(nbasis,1);

% Champagne-NL in the GSBF subspace, Eqs. (6)-(8), (12)-(14).
cost = nan(opts.max_iter+1,1);
[Z,z,q,cost(1)] = posterior(A,Y,gamma,lambda);
converged = false;
for iteration = 1:opts.max_iter
    gamma_new = sqrt(mean(Z.^2,2)./max(z,realmin));
    residual = Y - A*Z;
    lambda_new = max(sqrt(mean(residual.^2,2)./q),opts.noise_floor);

    change = max(norm(gamma_new-gamma)/max(norm(gamma),realmin), ...
                 norm(lambda_new-lambda)/max(norm(lambda),realmin));
    gamma = gamma_new;
    lambda = lambda_new;
    [Z,z,q,cost(iteration+1)] = posterior(A,Y,gamma,lambda);
    cost_change = abs(cost(iteration+1)-cost(iteration))/max(1,abs(cost(iteration)));
    if iteration >= 5 && cost_change < opts.tol && change < opts.tol
        converged = true;
        break
    end
end

% Recover source time courses in the original units.
Z = Z*(yscale/fscale);
source_est = W*Z;
[~,peak] = min(abs(tm));
result.time = tm;
result.meg_grad = meg_grad;
result.source_map = abs(source_est(:,peak));
result.source_magnitude = abs(source_est);
result.source_rms = sqrt(mean(source_est.^2,2));
result.gfp = struct('time',t,'meg',gm,'seeg',gs,'correlation',r);
result.seeg_statistics = struct('t',tstat,'p_raw',p, ...
    'p_bonferroni',p_bonf,'retained',contacts,'rank_weight',contact_weight);
result.basis = struct('Phi',W,'primary_nodes',primary,'centers',centers, ...
    'scales_mm',scales,'raw_alpha',alpha(centers));
result.coefficients = Z;
result.gamma = gamma*(yscale/fscale)^2;
result.noise_variance_whitened = lambda*yscale^2;
result.whitener = P;
result.predicted_meg = F*Z;
result.residual_meg = meg - result.predicted_meg;
result.fit = struct('iterations',iteration,'converged',converged, ...
    'cost_history',cost(1:iteration+1));
end

function [Z,z,q,cost] = posterior(A,Y,gamma,lambda)
C = (A.*gamma')*A' + diag(lambda);
R = chol((C + C')/2,'lower');
RY = R\Y;
RA = R\A;
Ri = R\eye(size(R,1));
Z = gamma.*(A'*(R'\RY));
z = sum(RA.^2,1)';
q = sum(Ri.^2,1)';
cost = 2*sum(log(diag(R))) + sum(RY(:).^2)/size(Y,2);
end
