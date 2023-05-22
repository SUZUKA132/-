filename = uigetfile('logo_csse.png');
img=imread(filename);
imshow(img);
img_gray = rgb2gray(img);
img_Brinarize = imbinarize(img_gray);
imshow(img_Brinarize*255)