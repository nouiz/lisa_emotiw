% Applique le preprocessing sur les images du dossier
function preprocessing_IS(path_in, path_out)

% Paramètres
%dirnames ={'C:\challenges\testimages\Unnamed\'}
dirnames = cellstr(path_in)
path_out=cellstr(path_out)
normalize = 0;
lambda = 7;

% Code
for dirname = dirnames
    dirname = dirname{1}
    dirname
    imagefiles = dir(strcat(dirname,'*.png'));     
    for i=1:length(imagefiles)
       currentfilename = strcat(path_in, imagefiles(i).name)
       currentimage = imread(currentfilename);
       %currentimage = imresize(currentimage, [96 96]);
       filename = strrep(currentfilename, path_in, path_out)
       output = isotropic_smoothing(currentimage,lambda,normalize)
       imwrite(output, filename{1});%,filename);
       %imwrite(currentimage, currentfilename);
    end
    disp('done')
end



