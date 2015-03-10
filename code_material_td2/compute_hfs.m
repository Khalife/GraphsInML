function [accuracy] = compute_hfs(graph_type, graph_thresh)
addpath('generate_data');
addpath('plot_graph');
%%%%%%%%%%%% the number of samples to generate
num_samples = 100;
N=num_samples;
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
n_l=70;
idxs = randperm(size(L,1),n_l);
l_idx = idxs; %pickup 4 random labels of samples
u_idx = 1:N; u_idx(idxs)=[];

C=zeros(size(X,1),size(X,1));
for i=1:size(X,1)
    if ismember(i,l_idx)
        C(i,i)=1;
    else
        C(i,i)=0.1;
    end 
end




Yl = zeros(length(l_idx),2);
for i=1:length(l_idx)
Yl(i,:) = [Y(l_idx(i)),3-Y(l_idx(i))];
end
Luu=L(u_idx,u_idx);
Lul=L(u_idx,l_idx);
Yu = Luu\(-Lul*Yl);

%Yl = zeros(length(Y),2);
%Yl= ;
%SYu = ;
gamma_g=0.01;
I=eye(size(L,1));
Q=L + gamma_g*I;
SYu= (C\Q+I)\[Yl;Yu];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the HFS solution       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,label] = max(Yu,[],2);
[~,soft_label] = max(SYu);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

final_label = Y;
final_label(u_idx) = label;

if plot_results
    plot_classification(X,Y,W,final_label, soft_label);
end

