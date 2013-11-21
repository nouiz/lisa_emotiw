function demoneimagewhole(folder_todo, folder_result, folder_done, model_no)


folder_todo='/u/kruegerd/rtest/';
folder_result='/u/kruegerd/rtest2/';
folder_done='/u/kruegerd/rtest3/';
model_no=2;


if nargin == 3
    model_no = 3;
end

%disp(model_no)

imagedir=dir(folder_todo);

% find first file (not a directory)
i = 1;
while imagedir(i).isdir
    i = i + 1;
end

path=[folder_todo imagedir(i).name];
path_res=[folder_result imagedir(i).name];
path_done=[folder_done imagedir(i).name];

% load and visualize model
switch model_no
    case 1
        % Pre-trained model with 146 parts. Works best for faces larger than 80*80
        load face_p146_small.mat
    case 2
        % Pre-trained model with 99 parts. Works best for faces larger than 150*150
        load face_p99.mat
    otherwise
        % Pre-trained model with 1050 parts. Give best performance on localization, but very slow
        load multipie_independent.mat
end

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
img=imread(path);

if size(img,3)==1
    im = repmat(img,[1 1 3]);
else
    im = img;
end

tic;
bs = detect(im, model, model.thresh);
bs = clipboxes(im, bs);
bs = nms_face(bs,0.3);
dettime = toc;

disp(size(bs))

if size(bs,1)>0
    % show highest scoring one
    [xs,ys]=showboxes(im, bs(1),posemap);title('Highest scoring detection');
    disp(transpose(xs))
    disp(transpose(ys))
    save(strcat(path_res(1:end-4),'.mat'),'xs','ys','bs','dettime');
    for kk=1:size(xs,2)
        im(int16(ys(kk)),int16(xs(kk)),:)=255;
    end
    disp(path_res)
    imwrite(im,path_res);
end
movefile(path,path_done);
end
