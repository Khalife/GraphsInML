function [W] = euclidean(X, similarity_options)
%[W] = euclidean(X, similarity_options)
% Computes euclidean distance between a set of samples.
% X is an n x m matrix of m-dimensional samples
% similarity_options is a vector containing options, but can be left empty for
% simple euclidean distance
% The return value W is an n x n dimensional matrix containing the
% distances between each pair of points.

    W = squareform(pdist(X,'euclidean'));
