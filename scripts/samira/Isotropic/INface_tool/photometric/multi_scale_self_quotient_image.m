% The function applies the multi scale self quotient image algorithm to an image.
% 
% PROTOTYPE
% Y=multi_scale_self_quotient_image(X,siz,sigma,normalize);
% 
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_self_quotient_image(X);
%       Y=normalize8(Y);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_self_quotient_image(X,[3 15 29]);
%       Y=normalize8(Y);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_self_quotient_image(X,[3 5 9 13 29], [0.5 1 2 0.2 4]);
%       Y=normalize8(Y);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_self_quotient_image(X,[3 5 9 13 29], [0.5 1 2 0.2 4], 0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% self quotient image technique. As described in the original paper the 
% technique is implemented with several scales (similar to 
% the multi scale retinex technique). This is an multi scale implementation
% of the technique in the "single_scale_self_quotient_image" function. If
% the length of the array size is 1 then this function returns the same
% result as the "single_scale_self_quotient_image".
% 
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. The default parameters are set as used in the chapter if teh
% AFIA book.
%
%
% 
% REFERENCES
% This function is an implementation of the multi scale self quotient
% image (MSSQI) algorithm proposed in:
%
% H. Wang, S.Z. Li, Y. Wang, and J. Zhang, “Self Quotient Image for Face
% Recognition,” in: Proc. of the International Conference on Pattern 
% Recognition, October 2004, pp. 1397- 1400.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% siz				    - an array of size 1 x n where n is the number of
%                         different filter sizes used; all entries in the
%                         array "siz" should be odd, the default values for
%                         the "siz" array are "siz = [3 5 11 15]"
% sigma                 - an array of the same size as "siz" defining the
%                         bandwidths of the isotropic base filters of the 
%                         corresponding filter scales; default 
%                         "sigma = [1 1.2 1.4 1.6]" 
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval) of the individual 
%                                single scale self quotient images - default
% 
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the multi scale
%                         self quotient image technique
%                         
%
% NOTES / COMMENTS
% This function applies the multi scale self quotient image technique to the
% grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate  - a function provided by Peter Kovesi
% normalize8    - auxilary function
% 
% ABOUT
% Created:        24.8.2009
% Last Update:    12.10.2011
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
function Y=multi_scale_self_quotient_image(X,siz,sigma,normalize);

%% Init 
Y=[];

%% Parameter checking
if nargin == 1
    siz = [3 5 11 15];
    sigma = [1 1.2 1.4 1.6];
    normalize = 1;
elseif nargin == 2
    t = length(siz);
    sigma = ones(1,t);
    normalize = 1;
elseif nargin == 3
    normalize = 1;
elseif nargin == 4
    if ~(normalize==1 || normalize==0)
        disp('Error: The fourth parameter can only be 0 or 1.');
        return;
    end   
elseif nargin > 4
    disp('Wrong number of input parameters.')
    return;
end

[a,b]=size(siz);
if a ~= 1 && b ~= 1
    disp('The parameter "siz" should be an 1 x n array.')
    return;
end

for i=1:length(siz)
   if mod(siz(i),2)==0
       disp('Error! All values in the input array "siz" need to be odd.')
       return;
   end
end

if length(siz) ~= length(sigma)
    disp('Error! The input arrays "siz" and "sigma" need to be of same size!')
    return;
end

%% Init. operations
X=normalize8(X);
[a,b]=size(X);


%% Apply the MSSQI
Y=zeros(a,b);
for i=1:length(siz)
    Y=Y+single_scale_self_quotient_image(X,siz(i),sigma(i),normalize);   
end

if normalize ~= 0
    Y=normalize8(Y);
end
































