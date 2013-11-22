% The function applies the single scale retinex algorithm to an image.
% 
% PROTOTYPE
% [R,L] = single_scale_retinex(X,hsiz, normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       R=single_scale_retinex(X);
%       figure,imshow(X);
%       figure,imshow((R),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       R=single_scale_retinex(X,9);
%       figure,imshow(X);
%       figure,imshow((R),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       R=single_scale_retinex(X,9,0);
%       figure,imshow(X);
%       figure,imshow((R),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       R=single_scale_retinex(X,[],0);
%       figure,imshow(X);
%       figure,imshow((R),[]);
% 
%     Example 5:
%       X=imread('sample_image.bmp');
%       [R,L] = single_scale_retinex(X,[],1);
%       figure,imshow(X);
%       figure,imshow((R),[]);
%       figure,imshow(L,[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% SSR technique. It takes either one, two or three arguments with the first being
% the image to be normalized, the second being the size of the Gaussian
% smoothing filter and the third being a parameter controling whether postprocessing
% (i.e., truncation of the histogram ends) is performed or not (0 - no 
% postprocessing, 1- truncation of the histogram ends and 
% range adjustment to the 8 bit interval). If no parameter "hsiz" is specified 
% or the parameter is specified in the form of empty brackets a default 
% value of hsiz=15 is used.
% 
% The function returns the photometricaly normalized form of the input
% image X in the output R. Since the technique is based on the retinex
% theory (image = reflectance*luminance), the function can optionally
% return the estimated luminance function in L as well.
%
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative.
%
% When studying the original paper of Jabson et al. one should be aware
% that "hsiz" corresponds to the parameter "c".
%
% 
% REFERENCES
% This function is an implementation of the Center/Surround retinex
% algorithm proposed in:
%
% D.J. Jobson, Z. Rahman, and G.A. Woodell, “Properties and Performance of a
% Center/Surround Retinex,” IEEE Transactions on Image Processing, vol. 6, 
% no. 3, pp. 451-462, March 1997.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% hsiz				    - a size parameter determining the size of the
%                         Gaussian filter (default: hsiz=15), hsiz is a
%                         scalar value; if [] is specified for hsiz the
%                         default value is used
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval) - default 
%
% OUTPUTS:
% R                     - a grey-scale image processed with the SSR
%                         algorithm (this normalized version of the input 
%                         image is commonly referred to as the reflectance)
% L                     - the luminance function estimated from the input
%                         grey-scale image X using the SSR tecnique
%                        
%
% NOTES / COMMENTS
% This function applies the single scale retinex algorithm on the
% grey-scale image X. In the new version of the function the input argument
% "normalize" was added to enable experimentation with ones own
% post-processors. In any case, if you do not have your own posprocessor, I
% would strongly recomend that you leave the parameter set to 1, as
% post-processing is often crucial for the normalization. 
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
% Created:        19.8.2009
% Last Update:    7.10.2011
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
function [R,L] = single_scale_retinex(X,hsiz, normalize)


%% Default result
R=[];
L=[];

%% Parameter checking
if nargin == 1
    hsiz = 15;
    normalize = 1;
elseif nargin ==2
    if isempty(hsiz)
        hsiz = 15;
    end
    normalize = 1;
elseif nargin == 3
    if isempty(hsiz)
        hsiz = 15;
    end
    
    if ~(normalize==1 || normalize==0)
        disp('Error: The third parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 3
    disp('Error: Wrong number of input parameters!')
    return;
end

%% Init. operations
[a,b]=size(X);
cent = ceil(a/2);
X1=normalize8(X)+0.01; %for the log operation

%% Filter construction
filt = zeros(a,b);
summ=0;
for i=1:a
    for j=1:b
        radius = ((cent-i)^2+(cent-j)^2);
        filt(i,j) = exp(-(radius/(hsiz^2)));
        summ=summ+filt(i,j);
    end
end
filt=filt/summ;

%% Filter image and adjust for log operation 
Z = ceil(imfilter(X1,filt,'replicate','same'));
if(sum(sum(Z==0))~=0)
    for i=1:a
        for j=1:b
            if Z(i,j)==0;
                Z(i,j)=0.01;
            end
        end
    end
end

%% Produce illumination normalized image Y
R=log(X1)-log(Z);
L=log(Z);

%% Do some postprocessing - this step is not necessary and can be skipped
if normalize ~= 0
    [R, dummy] =histtruncate(R, 0.2, 0.2);
    R=normalize8(R);
    L=normalize8(L);
end
   



















