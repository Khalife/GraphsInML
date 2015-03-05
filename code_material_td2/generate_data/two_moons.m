function [X, Y] = blobs(num_samples,dist_options)
%dist_options: radius of the moons, variance of the moons

    moon_radius = dist_options(1);
    moon_var = dist_options(2);
    moon_mislabel = dist_options(3);

    if mod(num_samples, 2) ~= 0
        error('The number of samples must be a multiple of the number of blobs');
    end

    X=zeros(num_samples,2);

    for i=1:num_samples/2
        r = moon_radius + 4 * (i-1)/num_samples;
        t = (i - 1) * 3/num_samples * pi;
        X(i, 1) = r*cos(t);
        X(i, 2) = r*sin(t);
        X(i + num_samples/2, 1) = r*cos(t + pi);
        X(i + num_samples/2, 2) = r*sin(t + pi);
    end

    X= X + sqrt(moon_var) * randn(num_samples, 2);
    Y = [ ones(num_samples/2, 1); 2*ones(num_samples/2, 1)];

    flip_labels = rand(num_samples,1) <= moon_mislabel;

    Yo = Y;

    Yo(flip_labels & Y==1) = 2;
    Yo(flip_labels & Y==2) = 1;

    Y = Yo;


