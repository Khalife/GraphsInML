function [y] = compute_solution(t, centroids, nodes_to_centroids_map, centroids_to_nodes_map, labels)
        gamma_g = 1e-3;
        num_centroids = size(centroids,1);


        %compute the similarity matrix on the centroids

        qW = build_faces_graph();


        %compute the normalized laplacian

        sqrtD = ;
        qW = ;
        qL = ;
        regL = ;

        %compute the multiplicities
        v = zeros(length(labels),1);

        for i = 1:length(centroids_to_nodes_map)
            v(----------(i)) = sum(---------------- == -----------------(i));
        end

        % compute the labels of the nodes that are currently a
        % representative for a centroid
        sublabs = ;


        Yl = [sublabs(sublabs ~= 0) == -1, sublabs(sublabs ~= 0) == 1];

        % compute the SHFS solution

        Wul = ;
        Luu = ;
        Vuu = ;
        Yu = full(  );

        %compute the labels
        y = sublabs;
        y(sublabs == 0) = ;% +1 -1 labels

        fprintf('label %d confidence %.5f opposite %.5f time %d\n',y(find(centroids_to_nodes_map == t)),Yu(find(centroids_to_nodes_map(sublabs == 0) == t),1), Yu(find(centroids_to_nodes_map(sublabs == 0) == t),2),t);

        y = y(find(centroids_to_nodes_map == t));
