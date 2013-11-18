% The function computes the gradientfaces version of the input image.
% 
% PROTOTYPE
% Y = gradientfaces(X,sigma, normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = gradientfaces(X,0.8, 1);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = gradientfaces(X,[], 0);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = gradientfaces(X);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y = gradientfaces(X,1);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the 
% Gradientface approach. The functions takes either one, two or three input 
% arguments. The first argument is the input image, the second 
% argument "sigma" stands for the standard deviation of the Gaussian used in 
% the filtering operation of the first step of the technique. The third 
% parameter "normalize" is the parameter that controls whether the input 
% result is scaled to the 8-bit interval or not.
%
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. 
%
% 
% 
% REFERENCES
% This function is an implementation of the Gradientfaces technique 
% proposed in:
%
% T. Zhang, Y.Y. Tang, B. Fang, Z. Shang, X. Liu: Face Recognition Under
% Varying Illumination Usng Gradientfaces, IEEE Transaction on Image
% Processing, vol. 18, no. 11, November 2009, pp. 2599-2606.
% 
% 
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% sigma				    - the standard deviation of the Gaussian filter used in
%                         the smoothing step; default = 0.75
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (range 
%                                adjustment to the 8-bit interval) - default
%
% OUTPUTS:
% Y                     - a grey-scale image normalized using Gradientfaces
% 
%
% NOTES / COMMENTS
% This function applies the Gradientfaces appraoch to the grey-scale image X. 
% 
%
% 
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% normalize8            - auxilary function
% 
% Created:        19.8.2009
% Last Update:    11.10.2011
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
function Y = gradientfaces(X,sigma, normalize)

%% Default results
Y=[];

%% parameter checking
if nargin == 1
    sigma = 0.75;
    normalize = 1;
elseif nargin == 2
    if isempty(sigma)
        sigma = 0.75;
    end
    normalize = 1;
elseif nargin == 3
    if isempty(sigma)
        sigma = 0.75;
    end
     if ~(normalize==1 || normalize==0)
        disp('Error: the third parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 3
   disp('Error: the function takes at most three parameters.');
   return;
end



%% Init. operations
[a,b]=size(X);
X1=normalize8(X); 


%% Gaussian smoothing
F = fspecial('gaussian',2*ceil(3*sigma)+1,sigma);
XF  = imfilter(X1,F,'replicate','same');

%% Construct derivatives of Gaussians in x and y directions
Wx = floor((7/2)*sigma); 
if Wx < 1
   Wx = 1;
end

Wy = floor((7/2)*sigma); 
if Wy < 1
   Wy = 1;
end
[Xw,Yw]=meshgrid(-Wy:Wy,-Wx:Wx);

%build derivative filters
Gx = -2.*Xw.*exp(-(Xw.^2+Yw.^2)./(2*sigma^2));
Gy = -2.*Yw.*exp(-(Xw.^2+Yw.^2)./(2*sigma^2));

%% Compute gradientfaces 

% This gives the same results as the ones shown in the paper; 
% however, this is different from the theoretical explanation in the paper;
% the range of the output values in this case is [-pi, pi]; to get the
% range given in Eq. (8) of the paper, the commented line below should
% be uncommented
Y = atan2(imfilter(X1,Gy,'replicate','same'),imfilter(X1,Gx,'replicate','same'));
%Y = Y.*(Y>=0)+(2*pi*ones(size(Y))-Y.*(Y<0));

%% postprocessing
if normalize ~= 0
    Y=normalize8(Y);  
end
 

















