% The function applies a wavelet-based normalization algorithm to an image.
% 
% PROTOTYPE
% [Y]=wavelet_normalization(X,fak,wname,mode,normalize);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=wavelet_normalization(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=wavelet_normalization(X,1.3);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=wavelet_normalization(X,1.2,'sym1');
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y=wavelet_normalization(X,1.4,'haar','sym');
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 5:
%       X=imread('sample_image.bmp');
%       Y=wavelet_normalization(X,[],[],[],0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 6:
%       X=imread('sample_image.bmp');
%       Y=wavelet_normalization(X,1.4,'haar','sym',0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% a wevelet-based normalization. The function equalizes the histogram of
% the approximation coefficients matrix and emphasizes (by scaling) the
% detailed coefficient in the three directions. As a final step it performs
% an inverse wavelet transform to recover the normalized image.
% 
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. The default parameters are set as used in the chapter of the
% AFIA book.
%
%
% 
% REFERENCES
% This function is an implementation of the wavelet-based photometric 
% normalization technique proposed in:
%
% S. Du, and R. Ward, “Wavelet-based Illumination Normalization for Face
% Recognition,” in: Proc. of the IEEE International Conference on Image 
% Processing, ICIP’05, September 2005.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% fak                   - a scalar value determining the emphasiz of the
%                         detailed coefficients, default "fak=1.5"
% wname                 - a string determining the name of the wavelet to
%                         use, "wname" is the same as used in "dtw2", for a
%                         list of options for this variable please refer to
%                         Matlabs internal help on "dwt2", default 
%                         "wname='db1'"
% mode                  - a string determining the extension mode, for help
%                         on the parameter please type "help dwtmode" into
%                         Matlabs command prompt, default value 
%                         "mode = 'sp0'"
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (normalization 
%                                to the 8-bit interval) - default
% 
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the wavelet-based
%                         normalization technique
%                         
%
% NOTES / COMMENTS
% This function applies the wavelet-based normalization technique to the
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
% Created:        25.8.2009
% Last Update:    13.10.2011
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

function [Y]=wavelet_normalization(X,fak,wname,mode,normalize);

%% Init
Y=[];

%% Parameter Check
if nargin == 1
    fak = 1.5;
    wname = 'db1';
    mode = 'sp0';
    normalize=1;
elseif nargin == 2
    if isempty(fak)
        fak = 1.5;
    end
    wname = 'db1';
    mode = 'sp0';
    normalize=1;
elseif nargin == 3
    if isempty(fak)
        fak = 1.5;
    end
    if isempty(wname)
        wname = 'db1';
    end
    mode = 'sp0';
    normalize=1;
elseif nargin == 4
    if isempty(fak)
        fak = 1.5;
    end
    if isempty(wname)
        wname = 'db1';
    end
    if isempty(mode)
        mode = 'sp0';
    end
    normalize=1;
elseif nargin == 5
    if isempty(fak)
        fak = 1.5;
    end
    if isempty(wname)
        wname = 'db1';
    end
    if isempty(mode)
        mode = 'sp0';
    end
     if ~(normalize==1 || normalize==0)
        disp('Error: The fifth parameter can only be 0 or 1.');
        return;
    end      
elseif nargin > 5
    disp('Wrong number of input parameters!')
    return;
end

%% Init operations
X=normalize8(X);

%% Perform the wavlet transform and normalize
[cA,cH,cV,cD] = dwt2(X,wname,'mode',mode);
cA=histeq(uint8(normalize8(cA)));
cH=fak*cH;
cV=fak*cV;
cD=fak*cD;

%% Invert the transform
Y = (idwt2(cA,cH,cV,cD,wname,'mode',mode));

%% Do some post-processing (or not)
if normalize ~=0 
    Y=normalize8(histtruncate(normalize8(Y),2,2));
end

































