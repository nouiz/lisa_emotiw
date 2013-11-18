%RANK_NORMALIZATION The function performs rank normalization on an image - histogram equalization.
% 
% PROTOTYPE
% Y=rank_normalization(X,mode,updown)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=rank_normalization(X);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=rank_normalization(X,'two');
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=rank_normalization(X,'three','descend');
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%
%
% GENERAL DESCRIPTION
% The function performs rank normalization on an image. This means that all
% pixels in an image are ordered from the most negative to the most
% positive (from the one with the smallest intensity value to the one with 
% the largest intensity valuel). After the ordering the first pixel is
% assigned a rank of one, the second the rank of two, ..., and the last is assigned
% a rank of N, where N is the number of pixels in the image. This procedure
% is identitcal to histogram equalization, except for the inteval to which
% the intensity values are mapped to. (For options on the interval take a 
% look at the description of the parameter "mode"). Unlike Matlabs internal
% function "histeq" this function also works with doubles, works faster and
% provides more flexibility regarding the output range of the pixel
% intensity values.
% 
%
%
% 
% REFERENCES
% This function is an implementation of the rank normalization for which a
% more detailed descriptions can be found in:
%
% Štruc, V., Žibert, J. in Pavešiæ, N.: Histogram remapping as a
% preprocessing step for robust face recognition, WSEAS transactions on 
% information science and applications, vol. 6, no. 3, pp. 520-529, 2009.
% (BibTex available from:
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/WSEAS.bib)
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% mode                  - a string determining the range of the output
%                         intensity values, valid options:
%                           'one'   - mapps output to [0 1] (default),
%                           'two'   - mapps output to [0 N], where N denotes 
%                                     the total number of pixels in X
%                           'three' - mapps output to [0 255] 
% updown                - a string defining the sort operation, valid
%                         options:
%                           'ascend'  - the output image is the histogram
%                                       equalized version of the input 
%                                       image
%                           'descend' - the output image is the histogram
%                                       equalized version of the inverted 
%                                       input image
% 
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the anisotropic
%                         smoothing
%                         
%
% NOTES / COMMENTS
% This function performs rank normalization on the intensity values of the
% grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% normalize8    - auxilary function
% 
% ABOUT
% Created:        26.8.2009
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
function Y=rank_normalization(X,mode,updown)
%% Default result
Y=[];

%% Parameter check
if nargin == 1
    mode = 'one';
    updown='ascend';
elseif nargin==2
    updown='ascend';
elseif nargin >3
    disp('Error: Wrong number of input parameters.')
    return;
end

if strcmp(updown,'ascend') == 0 && strcmp(updown,'descend')==0
    disp('Error: Wrong value for parameter updown.')
    return;
end

%% Init. operations
X=normalize8(X);
[n_rows,n_cols]=size(X);
N = n_rows*n_cols;
X = reshape(X,N,1);

%% Do the sorting and normalize
[Y,ind] = sort(X,updown);
for i=1:N
	Y(ind(i)) = i;
end
Y = reshape(Y,n_rows,n_cols);

%% Set output values
if strcmp(mode, 'one')%output [0,1]
	Y = normalize8(Y,0);
elseif strcmp(mode, 'two')%output [1 N]
    Y=Y;
elseif strcmp(mode, 'three')%output [0 255]
	Y = normalize8(Y,1);
else 
    disp('Wrong value for parameter "mode", Using "mode = one".')
    mode = 'one';
    Y=rank_normalization(X,mode);
end








