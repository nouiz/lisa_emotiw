function mapTFD2ICML(path1, path_out)


indx = [1:51];
ICMLavg5 = 'avgpointsICML5x.mat';
load(ICMLavg5); base_points = avg(indx,:)/5;

TFDavg = [path_out(1:end-1) '.mat'];%'avgpointsmissingvideos.mat';
load(TFDavg); input_points = avg(indx,:)+32;%add border

TFORM = cp2tform(input_points, base_points, 'similarity');
path1
imgdir=path1
imagefiles = dir([path1 '*.png'])
for i=1:length(imagefiles)
    i
    noisyim = uint8(randi([0,255],[160,160]));
    %[imgdir imagefiles(i).name]
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
    imwrite(imgtransformed,[path_out imagefiles(i).name],'png')
end
%verify
%ICMLavg5 = 'avgpointsICML5x.mat';
%load(ICMLavg5); base_points = avg(indx,:)/5;

%imagefiles = dir([savedir '*.png']);
%for i=1:10:length(imagefiles)
%    i
%    img = imread([savedir imagefiles(i).name]);
%    clf;
%   imshow(img)
%    hold on;
%   plot(base_points(:,1),base_points(:,2),'.r');
%   pause(0.0001)
%end
end
