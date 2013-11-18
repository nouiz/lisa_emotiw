function mapTFD2ICML()
ccc
imgdir = 'C:\challenges\FaceTubes\missingtestvideos\Done\';
%'C:\challenges\TFD\images96x96\';
savedir = 'C:\challenges\FaceTubes\missingtestvideos\Registered\';
%'C:\challenges\TFD\images96_transformed48\';
try
    rmdir(savedir,'s'); mkdir(savedir);
catch
end
indx = [1:51];
ICMLavg5 = 'avgpointsICML5x.mat';
load(ICMLavg5); base_points = avg(indx,:)/5;

TFDavg = 'avgpointsmissingvideos.mat';
load(TFDavg); input_points = avg(indx,:)+32;%add border

TFORM = cp2tform(input_points, base_points, 'similarity')

imagefiles = dir([imgdir '*.png']);
for i=1:length(imagefiles)
    i
    noisyim = uint8(randi([0,255],[160,160]));
    img = imread([imgdir imagefiles(i).name]);
    if size(img,3)==3
        img = rgb2gray(img);
    end
    x=(size(noisyim,1)-size(img,1))/2;
    y=(size(noisyim,2)-size(img,2))/2;
    noisyim(x+1:x+size(img,1),y+1:y+size(img,2))=...
        img;
    
    [imgtransformed,xdata,ydata] = imtransform(noisyim,TFORM,...
        'XData',[1 48], 'YData',[1 48]);
    %imshow(imgtransformed)
    imwrite(imgtransformed,[savedir imagefiles(i).name],'png')
end
%verify
ICMLavg5 = 'avgpointsICML5x.mat';
load(ICMLavg5); base_points = avg(indx,:)/5;

imagefiles = dir([savedir '*.png']);
for i=1:10:length(imagefiles)
    i
    img = imread([savedir imagefiles(i).name]);
    clf;
    imshow(img)
    hold on;
    plot(base_points(:,1),base_points(:,2),'.r');
    pause(0.0001)
end
end