% The function performs gamma correction on the input image X
% 
% PROTOTYPE
%       Y=gamma_correction(X, in_interval, out_interval, gamma);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=gamma_correction(X, [0 1], [0 1], 0.2);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=gamma_correction(X, [1 255], [0 255], 0.4);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=gamma_correction(X, [], [0 255], 0.4);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% in_interval           - a two component vector defining the input
%                         interval from which gamma correction is applied;
%                         this term is used to enable a linear shift of the 
%                         input values, e.g., [0 1]. If the value of the 
%                         interval is given with empty brackets, i.e., [], 
%                         then no adjustment is performed.  
% out_interval          - a two component vector defining the output
%                         interval of the gamma correction; when  
% gamma                 - the gamma value; i.e., the image is transofrmed
%                         using: X.^gamma 
%
% OUTPUTS:
% Y                     - a gamma corrected grey-scale image with its dynamic range
%                         adjusted to span the interval defined by the parameter
%                         "out_interval", 
% 
%
% NOTES / COMMENTS
% This function is needed by a few other function actually performing
% photometric normalization. It gamma corrects the intensity values of the images
% depending on the value of the input parameter "gamma". The function is
% similar to Matlabs "imadjust" function, which, however, is only available
% in the Image Processing Toolbox.
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
% 
% 
% ABOUT
% Created:        19.8.2009
% Last Update:    26.9.2011
% Revision:       2.0
% 
%
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
% OR ANY PART OF IT, MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
%
% 1. Štruc V., Pavešiæ, N.:Photometric normalization techniques for illumination 
% invariance, in: Y.J. Zhang (Ed), Advances in Face Image Analysis: Techniques 
% and Technologies, IGI Global, pp. 279-300, 2011.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/IGI.bib)
% 
% 2. Štruc, V., Pavešiæ, N.: Gabor-based kernel-partial-least-squares 
% discrimination features for face recognition, Informatica (Vilnius), 
% vol. 20, no. 1, pp. 115-138, 2009.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/InforVI.bib)
% 
%
% Official website:
% If you have down-loaded the toolbox from any other location than the
% official website, plese check the following link to make sure that you
% have the most recent version:
% 
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/INFace/index.html
% 
% 
% Copyright (c) 2011 Vitomir Štruc
% Faculty of Electrical Engineering,
% University of Ljubljana, Slovenia
% http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files, to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
% 
% October 2011

function Y=gamma_correction(X, in_interval, out_interval, gamma);

%% default return value
Y=[];

%% Parameter check
if nargin~=4
   disp('Error: The function takes exactly four arguments.');
   return;
end

%% Init. operations
X=double(X);
[a,b]=size(X);

%% Map to input interval
if ~isempty(in_interval)
    if length(in_interval)==2
        X=adjust_range(X,in_interval);
    else
       disp('Error: Input interval needs to be a two-component vector.');
       return;
    end
end

%% Do gamma correction
X=X.^gamma;    


%% Map to output interval
if ~isempty(out_interval)
    if length(out_interval)==2
        Y=adjust_range(X,out_interval);
    else
       disp('Error: Output interval needs to be a two-component vector.');
       return;
    end
end


