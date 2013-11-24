% The function applies the multi scale retinex algorithm to an image.
% 
% PROTOTYPE
% Y = multi_scale_retinex(X,hsiz,normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_retinex(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_retinex(X,[7 15 21]);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
%
%      Example 3 (the same as the SSR):
%       X=imread('sample_image.bmp');
%       Y=multi_scale_retinex(X,[15]);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_retinex(X,[],0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 5:
%       X=imread('sample_image.bmp');
%       Y=multi_scale_retinex(X,[7 15 21],0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% MSR technique. It takes either one or two arguments with the first being
% the image to be normalized and the second being the sizes of the Gaussian
% smoothing filters. If no parameter "hsiz" is specified a default value of
% hsiz=[7 15 21] is used. 
%
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative.
%
% When studying the original paper of Jabson et al. one should be aware
% that "hsiz" corresponds to the parameters "c".
%
% 
% REFERENCES
% This function is an implementation of the Multiscale retinex
% algorithm proposed in:
%
% D.J. Jabson, Z. Rahmann, and G.A. Woodell, “A Multiscale Retinex for
% Bridging the Gap Between Color Images and the human Observations of 
% Scenes,” IEEE Transactions on Image Processing, vol. 6, no. 7, 
% pp. 897-1056, July 1997.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% hsiz				    - a size parameter determining the sizes of the
%                         Gaussian filter (default: hsiz=[7 15 21]), hsiz
%                         is an array of filter sizes - actually these
%                         sizes determine the bandwidth of the filters
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval for the single-scale normalizations) - default
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the SSR
%                         algorithm
%
% NOTES / COMMENTS
% This function applies the multi scale retinex algorithm on the
% grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate  - a function provided by Peter Kovesi
% normalize8    - auxilary function
% 
% ABOUT
% Created:        20.8.2009
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
function Y = multi_scale_retinex(X,hsiz,normalize)

%% Dummy answer
Y=[];

%% Parameter checking and initialization
if nargin == 1
    hsiz = [7 15 21];
    normalize=1;
elseif nargin == 2
    if isempty(hsiz)
        hsiz = [7 15 21];
    end
    normalize=1;
elseif nargin == 3
     if isempty(hsiz)
        hsiz = [7 15 21];
     end
    
     if ~(normalize==1 || normalize==0)
        disp('Error: The third parameter can only be 0 or 1.');
        return;
    end
elseif nargin > 3
    disp('Error: Wrong number of input parameters!')
    return;    
end

[a,b]=size(hsiz);
if a ~= 1 && b ~= 1
    disp('The parameter "hsiz" should be an 1 x n array.')
    return;
end

[a,b]=size(X);


%% Apply multi-scale retinex
Y=zeros(a,b);
for i=1:length(hsiz)
    Y = Y + single_scale_retinex(X,hsiz(1,i),normalize);
end

if normalize ~=0
    Y=normalize8(Y);
end














