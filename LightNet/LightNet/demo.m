% =========================================================================
%% Deep Lowlight Image Enhancement with Auto-Save %%
% =========================================================================
close all; clear all; clc

% Add necessary paths (if any additional functions are in subfolders)
addpath(genpath('.'));

% Set gamma correction parameter
r = 1.7;

%% Read Image
% Prompt user to select an image file
[fn, pn, fi] = uigetfile('*.bmp;*.jpg;*.png;*.tif', 'Select Image');
im = imread([pn fn]);
figure, imshow(im), title('Input Image');

% Convert image to double precision
IM = im2double(im);

% Get image size
[height, width, channel] = size(IM);

% Lowlight image
LB = IM;

%% Deep Enhancement
% Load the pre-trained model for deep enhancement
model = 'lowlight4layersfinal.mat';  % Pre-trained model file
im_b = LB;
load(model);

%% Apply Convolution Layers
% conv1
f1 = convolution(im_b, weights_conv1, biases_conv1);
% conv2
f2 = convolution(f1, weights_conv2, biases_conv2);
% conv3
f3 = convolution(f2, weights_conv3, biases_conv3);
% conv4
f4 = convolution1(f3, weights_conv4, biases_conv4);

% Apply gamma correction and adjust brightness
map = f4.^r;

%% Guided Filtering
% Refine the map using guided filtering
p = map;
batch_size = 33;  % Size of the local patches
eps = 10^-3;  % Regularization parameter
map = guidedfilter(IM(:,:,1), p, batch_size, eps);

%% Retinex Model Enhancement
% Enhance the image using Retinex model (color channel-wise)
new(:,:,1) = LB(:,:,1) ./ (map);  % Red channel enhancement
new(:,:,2) = LB(:,:,2) ./ (map);  % Green channel enhancement
new(:,:,3) = LB(:,:,3) ./ (map);  % Blue channel enhancement

% Display the enhanced image
figure, imshow(abs(new), []), title('Enhanced Image');

%% Auto-save the Enhanced Image
% Create output folder if it doesn't exist
output_folder = fullfile(pn, 'Enhanced_Images');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Define output file path and name
[~, name, ext] = fileparts(fn);  % Extract the name and extension of the input image
output_filename = fullfile(output_folder, [name '_enhanced' ext]);

% Save the enhanced image to the output folder
imwrite(abs(new), output_filename);

% Print a confirmation message with the saved file path
fprintf('Enhanced image saved to: %s\n', output_filename);
