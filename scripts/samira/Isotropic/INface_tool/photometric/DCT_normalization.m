% The function applies the DCT-based normalization algorithm to an image.
% 
% PROTOTYPE
% Y=DCT_normalization(X,numb,normalize);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=DCT_normalization(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=DCT_normalization(X,40);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=DCT_normalization(X,[], 0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       Y=DCT_normalization(X,100, 0);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% DCT-based normalization technique. The technique sets a predifend number
% of DCT coefficients to zero and hence removes some of the low-frequency
% information contained in the images. Since this low-frequency information
% is considered to be susceptible to illumination changes, the function
% performs some kind of normalization.
% 
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. 
%
%
% 
% REFERENCES
% This function is an implementation of the DCT-base photometric 
% normalization technique proposed in:
%
% W. Chen, M.J. Er, and S. Wu, “Illumination Compensation and normalization
% for Robust Face Recognition Using Discrete Cosine Transform in 
% Logarithmic Domain,” IEEE Transactions on Systems, Man and Cybernetics – 
% part B, vol. 36, no. 2, pp. 458-466, April 2006.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% numb                  - a scalar value determining the number of DCT
%                         coefficients to set to zero, default "numb=50"
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval) - default 
% 
%
% OUTPUTS:
% Y                     - a grey-scale image processed with the DCT-based
%                         normalization technique
%                         
%
% NOTES / COMMENTS
% This function applies the DCT-based normalization technique to the
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
function Y=DCT_normalization(X,numb, normalize);

%% Parameter checking
Y=[];%dummy
if nargin == 1
    numb = 50;
    normalize = 1;
elseif nargin == 2
    if isempty(numb)
        numb = 50;
    end
    normalize = 1;
elseif nargin == 3
    if isempty(numb)
        numb = 50;
    end
    
    if ~(normalize==1 || normalize==0)
        disp('Error: The third parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 3
    disp('Eror: Wrong number of input parameters.')
    return;
end

[a,b]=size(X);
if numb > a*b
    disp('Error! The number of DCT coeffcients to discard cannot be larger than the numbr of pixels in the image.')
    return;
end


%% Init. operations
X=normalize8(X);
[a,b]=size(X);
M=a;
N=b;
coors = do_zigzag(X);


%% Transform to logarithm and frequency domains
X=log(X+1);
X=normalize8(X);
means= mean(X(:))+10; %we chose a mean near the true mean (the value +10 can be changed)
Dc = dct2(X); 

%% apply the normalization
c_11=log(means)*sqrt(M*N);
Dc(1,1)=c_11;
for i=2:numb+1
    ky = coors(1,i);
    kx = coors(2,i);
    Dc(ky,kx) = 0;
end
Y=(idct2(Dc));


%% Do some post-processing (or not)
if normalize ~=0
    Y=normalize8(histtruncate(normalize8(Y),0.2,0.2));
end




%% This function produces the zigzag coordinates
function output = do_zigzag(X);

%init operations
h = 1;
v = 1;
vmin = 1;
hmin = 1;
vmax = size(X, 1);
hmax = size(X, 2);
i = 1;
output = zeros(2, vmax * hmax);


%do the zigzag
while ((v <= vmax) & (h <= hmax))    
    if (mod(h + v, 2) == 0)                 
        if (v == vmin)       
            output(:,i) = [v;h];        
            if (h == hmax)
	      v = v + 1;
	    else
              h = h + 1;
            end;
            i = i + 1;
        elseif ((h == hmax) & (v < vmax))   
            output(:,i) = [v;h];  
            v = v + 1;
            i = i + 1;
        elseif ((v > vmin) & (h < hmax))    
            output(:,i) = [v;h];  
            v = v - 1;
            h = h + 1;
            i = i + 1;
        end       
    else                                    
       if ((v == vmax) & (h <= hmax))       
            output(:,i) = [v;h];  
            h = h + 1;
            i = i + 1;        
       elseif (h == hmin)                   
            output(:,i) = [v;h];  
            if (v == vmax)
	      h = h + 1;
	    else
              v = v + 1;
            end;
            i = i + 1;
       elseif ((v < vmax) & (h > hmin))     
            output(:,i) = [v;h];  
            v = v + 1;
            h = h - 1;
            i = i + 1;
       end
    end
    if ((v == vmax) & (h == hmax))          
        output(:,i) = [v;h];  
        break
    end

end

































