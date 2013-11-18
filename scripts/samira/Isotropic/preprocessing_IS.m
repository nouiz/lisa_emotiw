% Applique le preprocessing sur les images du dossier

% Paramètres
dirnames ={'C:\challenges\testimages\Unnamed\'}
normalize = 0;
lambda = 7;

% Code
for dirname = dirnames
    dirname = dirname{1}
    imagefiles = dir(strcat(dirname,'*.jpg'));     
    for i=1:length(imagefiles)
       currentfilename = strcat(dirname,imagefiles(i).name);
       currentimage = rgb2gray(imread(currentfilename));
       %currentimage = imresize(currentimage, [96 96]);
       imwrite(isotropic_smoothing(currentimage,lambda,normalize),...
           strrep(currentfilename, '.jpg', '_is.jpg'));
       imwrite(currentimage,...
           currentfilename);
    end
    disp('done')
end


