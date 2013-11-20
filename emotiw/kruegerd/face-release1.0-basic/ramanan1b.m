function v = ramanan1(im, model_no)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP
%
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
model.thresh = min(-0.65, model.thresh);
% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end
% END SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


bs = detect(im, model, model.thresh);
bs = clipboxes(im, bs);
bs = nms_face(bs, 0.3);

%highest scoring one
[xs,ys] = xyboxes(im, bs(1), posemap);
v = vertcat(xs,ys);

v = 3;

% all
% [xs,ys] = xyboxes(im, bs,posemap)

end
