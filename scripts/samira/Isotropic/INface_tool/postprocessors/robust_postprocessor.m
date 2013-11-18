% The function performs postprocessing of the photometrically normalized image X
% 
% PROTOTYPE
%       Y=robust_postprocessor(X, alfa, tao);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=robust_postprocessor(X, 0.1, 10);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=robust_postprocessor(X, 0.3);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=robust_postprocessor(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%
% REFERENCES
% This function is an implementation of the robust postprocessing step 
% proposed in the following publication under the Contrast Equalization 
% section:
%
% X. Tan and B.Triggs, Enhanced Local Texture Feature Sets for Face
% Recognition Under Difficult Lighting Conditions, IEEE Transactions on
% Image Processing, vol. 19, no. 6, June 2010, pp. 1635-1650.
% 
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% alfa                  - a parameter controlling the postprocessing
%                         procedure; default: 0.1
% tao                   - a parameter controlling the postprocessing
%                         procedure; default: 10 
%
% OUTPUTS:
% Y                     - the postprocessed image, where the global
%                         contrast has been normalized
% 
%
% NOTES / COMMENTS
% This function performs robust postprocessing of an image that has already 
% been photometrically normalized. The post-processing step is needed, 
% since images with different levels of illumination variation have 
% different contrasts after the normalization. In the prvious version of the 
% toolbox the upper and lower end of the histogram were simply truncated. 
% In this version, a second normalization procedure was added. This procedure
% is based on the work of Triggs and Tan. Note that the parameters "tao"
% and "alfa" are taken from the paper given in the reference section of
% this help. Please refer to the paper for more information on the
% normalization procedure and the meaning of the parameters.
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

function Y=robust_postprocessor(X, alfa, tao);

%% default return value
Y=[];

%% Parameter check
if nargin == 1
    alfa = 0.1;
    tao  = 10;
elseif nargin == 2
    tao = 10;
elseif nargin >3
   disp('Error: The function takes at most three arguments.');
   return;
end

%% Init. operations
X=double(X);
[a,b]=size(X);

%% Two-stage normalization
X = X/((mean((abs(X(:))).^alfa))^(1/alfa));
X = X/(( mean((min([tao*ones(1,a*b);abs(X(:)')])).^alfa) )^(1/alfa));

%% Nonlinear mapping
Y = tao*tanh(X/tao);


