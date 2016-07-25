rem Create the imagenet lmdb inputs
rem N.B. set the path to the imagenet train + val data dirs

set EXAMPLE=.\
set DATA=.\data
set BUILD=..\..\bin_cuda

set TRAIN_DATA_ROOT=%DATA%\classifier_training_images\
set VAL_DATA_ROOT=%DATA%\classifier_testing_images\

rem Set RESIZE=true to resize the images to 256x256. Leave as false if images have
rem already been resized using another tool.

set RESIZE_HEIGHT=114
set RESIZE_WIDTH=114



echo "Creating train lmdb..."

rd \s \q %EXAMPLE%\ocr_recog_train_lmdb
rd \s \q %EXAMPLE%\ocr_recog_test_lmdb

set GLOG_logtostderr=1 
%BUILD%\convert_imageset.cpp --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --shuffle --gray %TRAIN_DATA_ROOT% %DATA%\classifier_train_list.txt %EXAMPLE%\ocr_recog_train_lmdb

echo "Creating val lmdb..."

set GLOG_logtostderr=1 
%BUILD%\convert_imageset.cpp --resize_height=%RESIZE_HEIGHT% --resize_width=%RESIZE_WIDTH% --shuffle --gray %VAL_DATA_ROOT% %DATA%\classifier_test_list.txt %EXAMPLE%\ocr_recog_test_lmdb

echo "Done."
