function demoneimagewhole_scaled(path_in,path_out)
%path_in='C:\challenges\FaceTubes\missingtestvideos\DoneMissingtestvideos\';
%path_out='C:\challenges\FaceTubes\missingtestvideos\test\';
'scaled'

imagedir=dir(fullfile([path_in '*.png']));
%path_=['images/' imagedir(kk).name];
%path_res=['Result/' imagedir(kk).name];
%path_done=['Done/' imagedir(kk).name];

%compile;
% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
% load face_p146_small.mat

% % Pre-trained model with 99 parts. Works best for faces larger than 150*150
% load face_p99.mat

% % Pre-trained model with 1050 parts. Give best performance on localization, but very slow
load multipie_independent.mat

disp('Model visualization');
%visualizemodel(model,1:13);

% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-0.8, model.thresh);
% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

for i=1:length(imagedir)
    k=imagedir(i).name;
    img=imread([path_in k]);
    
    if size(img,3)==1
        im = repmat(img,[1 1 3]);
    else
        im = img;
    end
    
    imsize = size(im);
    %scl =  250/min(imsize(1),imsize(2))
    scl = 2;
    im = imresize(im,scl);
    size(im)
    
    tic;
    bs = detect(im, model, model.thresh);
    bs = clipboxes(im, bs);
    bs = nms_face(bs,0.3);
    dettime = toc;
    
    if size(bs,1)>0
        % show highest scoring one
        [xs,ys]=showboxes(im, bs(1),posemap);title('Highest scoring detection');
        xs = xs/scl; ys = ys/scl;
        path_res=[path_out k];
        save(strcat(path_res(1:end-4),'.mat'),'xs','ys','bs','dettime','posemap');
        for kk=1:size(xs,2)
            img(int16(ys(kk)),int16(xs(kk)),:)=255;
        end
        imwrite(img,[path_out k]);
        % show all
        %figure,showboxes(im, bs,posemap);title('All detections above the threshold');
        fprintf('Detection took %.1f seconds\n',dettime);
        close all;
    end
    %movefile(path_,path_done);
end
disp('done!');
end
