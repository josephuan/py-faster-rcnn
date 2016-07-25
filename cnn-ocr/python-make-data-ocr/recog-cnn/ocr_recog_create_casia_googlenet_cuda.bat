rem Create the imagenet lmdb inputs
rem N.B. set the path to the imagenet train + val data dirs

set EXAMPLE=.\
set DATA=d:\
set BUILD=..\..\..\..\bin_cuda

set TRAIN_DATA_ROOT=%DATA%
set VAL_DATA_ROOT=%DATA%

rem Set RESIZE=true to resize the images to 256x256. Leave as false if images have
rem already been resized using another tool.

set RESIZE_HEIGHT=114
set RESIZE_WIDTH=114



echo "Creating train lmdb..."

rd \s \q %EXAMPLE%\ocr_recog_casia_googlenet_train_lmdb
rd \s \q %EXAMPLE%\ocr_recog_casia_googlenet_test_lmdb

set GLOG_logtostderr=1 
%BUILD%\convert_imageset.cpp --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --shuffle --gray %TRAIN_DATA_ROOT% %EXAMPLE%\classifier_train_list.txt %EXAMPLE%\ocr_recog_casia_googlenet_train_lmdb

echo "Creating val lmdb..."

set GLOG_logtostderr=1 
%BUILD%\convert_imageset.cpp --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --shuffle --gray %VAL_DATA_ROOT% %EXAMPLE%\classifier_test_list.txt %EXAMPLE%\ocr_recog_casia_googlenet_test_lmdb

echo "Done."
