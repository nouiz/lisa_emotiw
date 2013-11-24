% The function adjusts the dynamic range of the grey scale image to the interval [0,255] or [0,1]
% 
% PROTOTYPE
%       Y=normalize8(X,mode);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=normalize8(X);
%       figure,imshow(X);
%       figure,imshow(uint8(Y));
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=normalize8(X,1);
%       figure,imshow(X);
%       figure,imshow(uint8(Y));
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=normalize8(X,0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% mode                  - the parameter indicates the range of the target
%                         interval, if "mode=1", the output is mapped to [0,
%                         255], if "mode=0" the output is mapped to [0,1]
%
% OUTPUTS:
% Y                     - a grey-scale image with its dynamic range
%                         adjusted to span the entire 8-bit interval, i.e.,
%                         the intensity values lie in the range [0 255], or
%                         the interval [0,1]
%
% NOTES / COMMENTS
% This function is needed by a few other function actually performing
% photometric normalization. It remapps the intensity values of the images
% from its original span to the interval [0, 255] or [0,1] depending on the 
% value of the input parameter "mode". If the parameter mode is not
% provided it is assumed that the target interval equals [0,255].
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
% 
% 
% RELATED FUNCTIONS (SEE ALSO) 
% histtruncate  - a function provided by Peter Kovesi
% adjust_range  - auxilary function
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

function Y=normalize8(X,mode);

%% default return value
Y=[];

%% Parameter check
if nargin==1
    mode = 1;
end

%% Init. operations
X=double(X);
[a,b]=size(X);

%% Adjust the dynamic range to the 8-bit interval
max_v_x = max(max(X));
min_v_x = min(min(X));

if mode == 1
    Y=ceil(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b))))*255);
elseif mode == 0
    Y=(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b)))));
else
    disp('Error: Wrong value of parameter "mode". Please provide either 0 or 1.')
end
    



