% The function adjusts the dynamic range of the grey scale image to a new
% user-defined interval given by "interval".
% 
% PROTOTYPE
%       Y=adjust_range(X,interval);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=adjust_range(X);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=adjust_range(X,[0,1]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=adjust_range(X,[0;0.1]);
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% interval              - a two-component vector defining the output range
%                         of the pixel intensities; i.e., the intreval to 
%                         which the image pixels are mapped to, e.g., [0 255] 
%
% OUTPUTS:
% Y                     - a grey-scale image with its dynamic range
%                         adjusted to span the desired interval
%
% NOTES / COMMENTS
% This function is needed by a few other functions actually performing
% photometric normalization. It remapps the intensity values of the images
% from its original span to the desired interval depending on the 
% value of the input parameter "interval".
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
% 
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate  - a function provided by Peter Kovesi
% normalize8    - auxilary function
% 
% 
% ABOUT
% Created:        19.8.2009
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

function Y=adjust_range(X,interval);

%% default return value
Y=[];

%% Parameter check
if nargin==1
    interval = [0; 255];
elseif nargin == 2
    if length(interval)~= 2
       disp('Error: The argument interval must be a two-component vector.')
       return;
    end
    
    if(interval(1)>interval(2))
        disp('Error: The argument "interval" does not specify a valid interval.');
        return;
    end
else
   disp('Error: The function takes at most two arguments.');
   return;
end

%% Init. operations
X=double(X);
[a,b]=size(X);
min_new = interval(1);
max_new = interval(2);

%% Adjust the dynamic range 
max_old = max(max(X));
min_old = min(min(X));

Y = ((max_new - min_new)/(max_old-min_old))*(X-min_old)+min_new;


