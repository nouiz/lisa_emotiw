% The function applies the Tan and Triggs normalization technique to an image
% 
% PROTOTYPE
% Y = tantriggs(X,gamma, normalize);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = tantriggs(X);
%       figure,imshow(X);
%       figure,imshow(uint8(Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = tantriggs(X,0.7,0);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = tantriggs(X,0.3);
%       figure,imshow(X);
%       figure,imshow(uint8(Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y = tantriggs(X,[],0);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% Tan and Triggs technique. The function takes wither one, two or three 
% input arguments, where the first is the image to be normalized, the 
% second is the gamma parameter relating to gamma intensity correction and 
% the third is a parameter controling whether a subsequent normalization of
% the photometrically normaliued image is performed or not (whether the 
% dynamic range of the pixel intensities is scaled to the 8-bit interval or 
% not). 
% 
% If no input arguments are provided, the function uses a default value
% for the gamma parameter (gamma = 0.2) and performs the final range 
% adjustment. The parameters "alpha" and "tao" from the original paper of
% Tan and Triggs are left as suggested by the authors in their TIP paper.
%
%
% 
% REFERENCES
% This function is an implementation of the Tan and Triggs technique 
% described in:
%
% X. Tan and B. Triggs: Enhanced Local Texture Sets for Face Recognition
% Under Difficult Lighting Conditions, IEEE Transactions on Image
% Processing, vol. 19, no. 6, June 2010, pp. 1635-1650.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% gamma                 - a scalar value refered to as the gamma parameter
%                         that controls the gamma intensity correction;
%                         default: 0.2 (can be [] if default wants to be used)
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (range 
%                                adjustment to the 8-bit interval) - default 
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the Tan and
%                         Triggs technique
%                         
%
% NOTES / COMMENTS
% This function applies the Tan and Triggs photometric normalization 
% technique to the grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% normalize8    - auxilary function
% 
% ABOUT
% Created:        25.8.2009
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
function Y = tantriggs(X,gamma, normalize);

%% Default return
Y=[];

%% Parameter checking

if nargin == 1
    gamma = 0.2;
    normalize = 1;
elseif nargin == 2
    if isempty(gamma)
        gamma = 0.2; 
    end
    normalize = 1;  
elseif nargin == 3
    if isempty(gamma)
        gamma = 0.2; 
    end
    
    if ~(normalize==1 || normalize==0)
         disp('Error: The third parameter can only be 0 or 1.');
         return;
     end
elseif nargin >3
    disp('Error! Wrong number of input parameters.')
    return;
end

%% Init. operations
[a,b]=size(X);

%% Gamma correction 
% (we could use 255*imadjust(X,[],[0,1],gamma), but would add dependencies 
% to the image processing toolbox); we use our implementation
Y=gamma_correction(X, [0 1], [0 255], gamma);

%% Dog filtering
Y = dog(Y,1, 2, 0);

%% Postprocessing
Y=robust_postprocessor(Y);

%% Normalization to 8bits
if normalize ~= 0
    Y=normalize8(Y);  
end

   





