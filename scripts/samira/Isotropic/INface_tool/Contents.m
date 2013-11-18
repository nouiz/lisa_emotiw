% INFACE_TOOL
%
% Files
%   install_INface - This is the install script for the INface toolbox v2.0. 
%   check_install  - This scripts checks the installation of the INface toolbox v2.0. 
% 
% 
% /AUXILARY
%
% Files
%   compute_patch_library    - The function cumpets the patch library.
%   highboostfilter          - The function constructs a high-boost Butterworth filter.
%   highpassfilter           - The function constructs a high-pass butterworth filter.
%   lowpassfilter            - the function constructs a low-pass butterworth filter.
%   normalize8               - The function adjusts the dynamic range of the grey scale image to the interval [0,255] or [0,1]
%   pca                      - The function performs principal component analysis.
%   perform_lowdim_embedding - The function performs a patch wise dimension extension
%   perform_nl_means         - The function applies the non-local means algorithm to an image.
%   perform_nl_means_adap    - The function perfroms adaptive non-local means normalization to an image
%   symmetric_extension      - The function perform a symmetric extension of the signal.
%   adjust_range             - The function adjusts the dynamic range of the grey scale image to a new
%   gamma_correction         - The function performs gamma correction on the input image X
%   threshold_filtering      - The function performs threshold filtering of an image
% 
% 
% /DEMOS
%
% Files
%   combin_demo      - A demo script showing the application of the photometric normalization techniques in conjucntion with rank normalization on a sample image.
%   histograms_demo  - A demo script showing the application of the histogram manipulation techniques on a sample image.
%   photometric_demo - A demo script showing the application of the photometric normalization techniques on a sample image.
%   luminance_demo       - A demo script showing the procedure of computing the luminance functions.
%   make_new_method_demo - A demo script how to combine different functions from the toolbox into a novel technique
% 
% 
% /HISTOGRAMS
%
% Files
%   fitt_distribution  - The function fitts a predefined distribution to the histogram of an image.
%   rank_normalization - The function performs rank normalization on an image - histogram equalization.
%  
% 
% /PHOTOMETRIC
%
% Files
%   adaptive_nl_means_normalization  - The function applies the adaptive non-local means normalization technique to an image
%   adaptive_single_scale_retinex    - The function applies the adaptive single scale retinex algorithm to an image.
%   anisotropic_smoothing            - The function applies the anisotropic smoothing normalization technique to an image
%   DCT_normalization                - The function applies the DCT-based normalization algorithm to an image.
%   homomorphic                      - The function perfroms homomorphic filtering on an image.
%   isotropic_smoothing              - The function applies the isotropic smoothing normalization technique to an image
%   multi_scale_retinex              - The function applies the multi scale retinex algorithm to an image.
%   multi_scale_self_quotient_image  - The function applies the multi scale self quotient image algorithm to an image.
%   nl_means_normalization           - The function applies the non-local means normalization technique to an image
%   single_scale_retinex             - The function applies the single scale retinex algorithm to an image.
%   single_scale_self_quotient_image - The function applies the single scale self quotient image algorithm to an image.
%   steerable_gaussians              - The function normalizes an image using steerable derivatives of gaussians
%   wavelet_denoising                - The function applies the wavelet denoising normalization to an image.
%   wavelet_normalization            - The function applies a wavelet-based normalization algorithm to an image.
%   anisotropic_smoothing_stable     - The function applies a modified version of anisotropic smoothing to an image
%   dog                              - The function applies a DoG (Difference of Gaussians) filter to an image.
%   gradientfaces                    - The function computes the gradientfaces version of the input image.
%   lssf_norm                        - The function applies the large and small scale features appoach to an input image.
%   multi_scale_weberfaces           - The function computes the multi-scale Weberface version of the input image.
%   tantriggs                        - The function applies the Tan and Triggs normalization technique to an image
%   weberfaces                       - The function computes the Weberface version of the input image.
% 
% 
% /POSTPROCESSORS
%
% Files
%   histtruncate         - The function truncates the ends of an image histogram.
%   robust_postprocessor - The function performs postprocessing of the photometrically normalized image X



