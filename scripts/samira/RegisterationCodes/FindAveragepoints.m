function FindAveragepoints(path_in, path_out)
ccc
rootres = 'C:\challenges\FaceTubes\missingtestvideos\Result\';
savename = 'avgpointsmissingvideos.mat';

load('info.mat');
imagefiles = dir([rootres '*.png']);
avgpoints = zeros(68,2);
count = 0;

for i=1:length(imagefiles)
    i
    img = imread([rootres imagefiles(i).name]);
    load([rootres imagefiles(i).name(1:end-3) 'mat']);
    points = [xs; ys]';
    
    if 1
        clf;
        imshow(img)
        hold on;
        plot(points(:,1),points(:,2),'or');
        pause(0.0001)
    end
    
    if posemap(bs(1).c)==0
        avgpoints = avgpoints + points;
        count = count+1;
    end
end
avg = avgpoints/count;
save(savename,'avg');

%verify average point
for i=1:length(imagefiles)
    i
    img = imread([rootres imagefiles(i).name]);
    load(savename);
    
    clf;
    imshow(img)
    hold on;
    plot(avg(:,1),avg(:,2),'or');
    pause(0.0001)
end
end
