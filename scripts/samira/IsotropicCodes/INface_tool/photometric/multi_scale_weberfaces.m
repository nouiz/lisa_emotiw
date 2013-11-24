% The function computes the multi-scale Weberface version of the input image.
% 
% PROTOTYPE
% Y = multi_scale_weberfaces(X,sigma, nn, alfa, weights, normalize)
% 
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = multi_scale_weberfaces(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = multi_scale_weberfaces(X,[1 0.5], [9 49], [4 0.04], [0.5 1], 1);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = multi_scale_weberfaces(X,[1 0.25], [9 25], [2 0.4], [1 0.5], 0);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the 
% multi-scale Weberface approach. The functions takes either one or 
% six input arguments. The first srgument "sigma" stands for the vector of standard 
% deviations of the Gaussians used in the filtering operation of the first 
% step of the technique. The second parameter "nn" denotes the vector of sizes of the 
% local neigborhoods taken into account when computing the Weberfaces. Note 
% that this parameter has to be in the form: nn = (2*integer+1)^2, i.e.,
% 9,25,49,81,... The fourth parameter "alfa" is the vector of magnification
% parameters
% used for balancing the input range of the "atan" function and "normalize"
% is the parameter that controls whether the input result is scaled to the
% 8-bit interval or not. The multi-scale form of the Weberfaces appraoch is
% computed as a linear cobination of single-scale Weberfaces with different
% neigborhood sizes (nn). These single scale Weberfaces can be computed
% with dofferent std-s and are finally combined using the weights in the
% vector weights, i.e., 
% 
%           Y = weights(1)*Weberface(X,sigma(1),nn(1),alfa(1)) +
%             + weights(2)*Weberface(X,sigma(2),nn(2),alfa(2)) +
%             + ... +
%             + weights(N)*Weberface(X,sigma(N),nn(N),alfa(N))    
%
% The concept is similar to that used in the multi scale retinex or multi
% scale self quotient image.
% 
% 
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. 
%
% 
% 
% REFERENCES
% This function is a straight forward extension of the Weberface technique 
% proposed in:
%
% B. Wang, W. Li, W. Y, Q. Liao: Illumination Normalization Based on
% Weber's Law with Application to Face Recognition, IEEE Signal Processing
% Letters, vol. 18, no. 8, August 2011, pp. 462-465.
% 
% 
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% sigma				    - a vector with standard deviations of the Gaussian 
%                         filtesr used in the smoothing step; 
%                               default: sigma = [1 0.75 0.5];
% nn				    - the sizes of the neighborhoods used for computing
%                         the Weberfaces; 
%                               default: nn = [9 25 49];
% alfa                  - a vector of parameters balancing the range of the 
%                         input values of the atan function; 
%                               default: alfa = [2 0.2 0.02];
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (range 
%                                adjustment to the 8-bit interval) - default
%
% OUTPUTS:
% Y                     - a grey-scale image normalized using multi-scale 
%                         Weberfaces
% 
%
% NOTES / COMMENTS
% This function applies the multi-scale Weberfaces appraoch to the 
% grey-scale image X. 
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
function Y = multi_scale_weberfaces(X,sigma, nn, alfa, weights, normalize)

%% Default results
Y=[];

%% parameter checking
if nargin == 1
    sigma = [1 0.75 0.5];
    nn = [9 25 49];
    alfa = [2 0.2 0.02];
    weights = [1 1 1];
    normalize = 1;
elseif nargin ==6       
     if ~(normalize==1 || normalize==0)
        disp('Error: the sixth parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 6
   disp('Error: the function takes at most six parameters.');
   return;
end

number_of_scales = max([length(sigma), length(nn), length(alfa), length(weights)]);

if(length(sigma) ~= number_of_scales || length(nn)~=number_of_scales || length(alfa)~=number_of_scales || length(weights)~=number_of_scales)
    disp('Error: the vectors "sigma", "nn", "alfa", and "weights" must be of same length.');
    return;
end

 % check neighborhood size
for i = 1:number_of_scales 
    nn1=nn(i);
     if(sqrt(nn1)-ceil(sqrt(nn1))>1e-5)
        disp('Error: each element in the neighborhood size vector "nn" needs to be a squared odd number.')
        return;
     end
end



%% Init. operations
[a,b]=size(X);
X1=normalize8(X); 

%calculate needed padding
how_many = length(sigma);

%% Calculate all weberfaces
Y = zeros(a,b);
for i=1:how_many
   Y=Y+weights(i)*weberfaces(X1,sigma(i),nn(i),alfa(i),0); 
end


%% postprocessing
if normalize ~= 0
    Y=normalize8(Y);  
end
 

















