% The function applies a DoG (Difference of Gaussians) filter to an image.
% 
% PROTOTYPE
% Y = dog(X,sigma1, sigma2, normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = dog(X);
%       figure,imshow(X);
%       figure,imshow(uint8(Y));
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = dog(X,1, 2);
%       figure,imshow(X);
%       figure,imshow(uint8(Y));
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = dog(X,1, 2,0);
%       figure,imshow(X);
%       figure,imshow(uint8(Y));
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y = dog(X,[],[],1); % uses default values for sigma1 and sigma2
%       figure,imshow(X);
%       figure,imshow(uint8(Y));
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using DoG 
% filtering. It takes either one, two, three or four arguments as input. Here,
% the first argument denotes input image, the second argument denotes the std 
% of the smaller Gaussian, the third argument stands for the std of the larger 
% Gaussian, and the fourth argument controls the the post-procesing (0 - no 
% post-processing, 1 - trancation of the ends of the histogram and  
% range adjustment to the 8 bit interval). If any of the parameters is 
% skipped, the defauilt values are used. 
%
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. You should also be aware that DoG filtering will not result in
% good normalization results if no Log transform or gamma correction has
% been applied to the input image prior to the filtering procedure.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% sigma1				- the standard deviation of the smaller Gaussian of
%                         the DoG filter; default = 1
% sigma2				- the standard deviation of the larger Gaussian of
%                         the DoG filter; default = 2
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval) - default
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the DoG filter
% 
%
% NOTES / COMMENTS
% This function applies DoG filtering on the grey-scale image X. 
% 
% In the new version of the function the input argument
% "normalize" was added to enable experimentation with ones own
% post-processors. In any case, if you do not have your own post-processor, I
% would strongly recomend that you leave the parameter set to 1, as
% post-processing is often crucial for the normalization.
%
% 
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate          - a function provided by Peter Kovesi
% normalize8            - auxilary function
% robust_postprocessor  - auxilary function
% 
% ABOUT
% Created:        19.8.2009
% Last Update:    28.9.2011
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
function Y = dog(X,sigma1, sigma2, normalize)

%% Default results
Y=[];

%% Parameter checking
if nargin == 1
    sigma_one = 1;
    sigma_two = 2;
    normalize = 1;
elseif nargin == 2
    if isempty(sigma1)
        sigma1 = 1;
    end
    
    sigma_one = sigma1;
    sigma_two = 2*sigma1;
    normalize = 1;
elseif nargin == 3
    if isempty(sigma1)
        sigma1 = 1;
    end
    
    if isempty(sigma2)
       sigma2 = 2; 
    end
    
    if ~(length(sigma1)==1 && length(sigma2)==1)
       disp('Error: The parameters sigma1 and sigma2 need to be scalars.');
       return;
    else
        sigma_one = sigma1;
        sigma_two = sigma2;
    end 
    normalize = 1;
elseif nargin == 4   
    if isempty(sigma1)
        sigma1 = 1;
    end
    
    if isempty(sigma2)
       sigma2 = 2; 
    end
        sigma_one = sigma1;
        sigma_two = sigma2; 
        
     if ~(normalize==1 || normalize==0)
        disp('Error: The fourth parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 4
   disp('Error: The function takes at most four parameters.');
   return;
end



%% Init. operations
[a,b]=size(X);
F1 = fspecial('gaussian',2*ceil(3*sigma_one)+1,sigma_one);
F2 = fspecial('gaussian',2*ceil(3*sigma_two)+1,sigma_two);
X1=normalize8(X); 

%% Filtering
XF1  = (imfilter(X1,F1,'replicate','same'));
XF2  = (imfilter(X1,F2,'replicate','same'));

Y = XF1-XF2;

%% postprocessing
if normalize ~= 0
    [Y, dummy] =histtruncate(Y, 0.2, 0.2);
    Y=normalize8(Y);  
end
 

















