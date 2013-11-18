% The function computes the Weberface version of the input image.
% 
% PROTOTYPE
% Y = weberfaces(X,sigma, nn, alfa, normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = weberfaces(X,1, 9, 2, 0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = weberfaces(X);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = weberfaces(X,[], 25, 2);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y = weberfaces(X,[], [], [], 0);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the 
% Weberface approach. The functions takes either one, two, three, four or 
% five input arguments, where any of the arguments can be left out and 
% replaced with empty brackets, i.e. []. In this case, the default value of 
% the argument is used. The first argument is the input image, the second 
% argument "sigma" stands for the standard deviation of the Gaussian used in 
% the filtering operation of the first step of the technique. The third 
% parameter "nn" denotes the size of the local neigborhood taken into 
% account when computing the Weberfaces. Note that this parameter has to be 
% of the following form: nn = (2*integer+1)^2, i.e., 9,25,49,81,... 
% The fourth parameter "alfa" is the magnification parameter
% used for balancing the input range of the "atan" function and "normalize"
% is the parameter that controls whether the input result is scaled to the
% 8-bit interval or not.
%
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. 
%
% 
% 
% REFERENCES
% This function is an implementation of the Weberface technique proposed in:
%
% B. Wang, W. Li, W. Y, Q. Liao: Illumination Normalization Based on
% Weber's Law with Application to Face Recognition, IEEE Signal Processing
% Letters, vol. 18, no. 8, August 2011, pp. 462-465.
% 
% 
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% sigma				    - the standard deviation of the Gaussian filter used in
%                         the smoothing step; default = 1
% nn				    - the size of the neighborhood used for computing
%                         the Weberfaces; default = 9
% alfa                  - a parameter balancing the range of the input values 
%                         of the atan function; default = 2
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (range 
%                                adjustment to the 8-bit interval) - default
%
% OUTPUTS:
% Y                     - a grey-scale image normalized using Weberfaces
% 
%
% NOTES / COMMENTS
% This function applies the Weberfaces appraoch to the grey-scale image X. 
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
% Last Update:    10.10.2011
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
function Y = weberfaces(X,sigma, nn, alfa, normalize)

%% Default results
Y=[];

%% parameter checking
if nargin == 1
    sigma = 1;
    nn = 9;
    alfa = 2;
    normalize = 1;
elseif nargin == 2
    if isempty(sigma)
        sigma = 1;
    end
    
    nn = 9;
    alfa = 2;
    normalize = 1;
elseif nargin == 3
    if isempty(sigma)
        sigma = 1;
    end
    
    if isempty(nn)
        nn = 9;
    end
    
    alfa = 2;
    normalize = 1;
elseif nargin == 4   
    if isempty(sigma)
        sigma = 1;
    end
    
    if isempty(nn)
        nn = 9;
    end
    
    if isempty(alfa)
        alfa = 9;
    end
    normalize = 1;
elseif nargin ==5
     if isempty(sigma)
        sigma = 1;
    end
    
    if isempty(nn)
        nn = 9;
    end
    
    if isempty(alfa)
        alfa = 9;
    end
    
     if ~(normalize==1 || normalize==0)
        disp('Error: the fifth parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 5
   disp('Error: the function takes at most five parameters.');
   return;
end

 % check neighborhood size
 if(sqrt(nn)-ceil(sqrt(nn))>1e-5)
    disp('Error: neighborhood size "nn" needs to be a squared odd number.')
    return;
 end
 
 % check oddness
 if(mod(sqrt(nn),2)~=1)
    disp('Error: square root of neighborhood size "nn" needs to be an odd number.')
    return;
 end



%% Init. operations
[a,b]=size(X);
X1=normalize8(X); 

%calculate needed padding
in_one_dim = (sqrt(nn)-1)/2;

%% Gaussian smoothing
F = fspecial('gaussian',2*ceil(3*sigma)+1,sigma);
XF  = imfilter(X1,F,'replicate','same');


%% Weberface calulation

% padding
XFP = padarray(XF,[in_one_dim,in_one_dim],'symmetric','both');
Y = zeros(a,b);

% main loop
for i=in_one_dim+1:a+in_one_dim
    for j=in_one_dim+1:b+in_one_dim
        argument = sum(sum(     (XFP(i,j)-XFP(i-in_one_dim:i+in_one_dim,j-in_one_dim:j+in_one_dim))/(XFP(i,j)+0.01)           ));
        Y(i-in_one_dim, j-in_one_dim) = atan(alfa*argument);
    end
end


%% postprocessing
if normalize ~= 0
    Y=normalize8(Y);  
end
 

















