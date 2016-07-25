rem Compute the mean image from the imagenet training lmdb
rem N.B. this is available in data/ilsvrc12

set EXAMPLE=.\
set DATA=.\
set BUILD=..\..\..\..\bin_cuda

%BUILD%\compute_image_mean %EXAMPLE%\ocr_recog_casia_googlenet_64in_train_lmdb %DATA%\ocr_recog_casia_googlenet_64in_mean.binaryproto

echo "Done."
