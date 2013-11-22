function FindAveragepointsDK(path_in, path_out)
path_out=[path_out(1:end-1) '.mat'];
%path_in = 'C:\challenges\FaceTubes\missingtestvideos\Result\';
%path_out = 'C:\challenges\FaceTubes\missingtestvideos\avgpoint.mat';

load('info.mat');
imagefiles = dir([path_in '*.png']);
avgpoints = zeros(68,2);
count = 0;

for i=1:length(imagefiles)
    img = imread([path_in imagefiles(i).name]);
    load([path_in imagefiles(i).name(1:end-3) 'mat']);
    points = [xs; ys]';
    
    if posemap(bs(1).c)==0
        avgpoints = avgpoints + points;
        count = count+1;
    end
end
avg = avgpoints/count;
save(path_out,'avg');

end
