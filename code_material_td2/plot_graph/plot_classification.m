function [] = plot_classification(X,Y,W,spectral_labels,soft_labels)

    set(figure(), 'units', 'centimeters', 'pos', [0 0 30 10]);

    h(1) = subplot(1,3,1);
    plot_edges_and_points(X,Y,W,'ground truth');

    h(2) = subplot(1,3,2);
    plot_edges_and_points(X,spectral_labels,W,'CHFS');
    
    h(3) = subplot(1,3,3);
    plot_edges_and_points(X,soft_labels,W,'SHFS');

    %linkaxes(h,'y')

    %ylim([-2,2])
