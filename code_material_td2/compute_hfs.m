function [accuracy] = compute_hfs(graph_type, graph_thresh)
addpath('generate_data');
%%%%%%%%%%%% the number of samples to generate
num_samples = 100;

%%%%%%%%%%%% the sample distribution function with the options necessary for the
%%%%%%%%%%%% distribuion


sample_dist = @two_moons;
dist_options = [1, 0.02, 0.1]; % two moons: radius of the moons,
                          %        variance of the moons
                          %        number of mislabeled nodes

plot_results = 0;

if nargin < 1

    plot_results = 1;

    %%%%%%%%%%%% the type of the graph to build and the respective threshold

    %graph_type = 'knn';
    %graph_thresh = 7; % the number of neighbours for the graph

    graph_type = 'eps';
    graph_thresh = 0.15; % the epsilon threshold

end

%%%%%%%%%%%% similarity function
similarity_function = @exponential_euclidean;

%%%%%%%%%%%% similarity options
similarity_options = [0.5]; % exponential_euclidean: sigma

[X, Y] = get_samples(sample_dist, num_samples, dist_options);

%%%%%%%%%%%% automatically infer number of labels from samples
num_classes = length(unique(Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% randomly sample six labels                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the build_similarity_graph function to build the graph W  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W = build_similarity_graph(graph_type, graph_thresh, X, similarity_function, similarity_options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build the laplacian                                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = diag(sum(W,2));
L = D-W;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute hfs solution                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_l=4;
idxs = randi(size(L,1),n_l,1);
l_idx = idxs; %pickup 4 random labels of samples
u_idx = 1:N; u_idx(idxs)=[];

Yl = zeros(length(l_idx),2);
for i=1:length(l_idx)
Yl(i,:) = [Y(l_idxs(i)),1-Y(l_idxs(i))];
end
Luu=L(u_idx,u_idx);
Lul=L(u_idx,l_idx);
Yu = Luu\(-Lul*Yl);

%Yl = zeros(length(Y),2);
%Yl= ;
%SYu = ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the HFS solution       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [~,label] = ;
% [~,soft_label] = ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

final_label = Y;
final_label(u_idx) = label;

if plot_results
    plot_classification(X,Y,W,final_label, soft_label);
end

