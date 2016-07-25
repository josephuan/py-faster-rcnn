set TOOLS=..\..\bin_cuda

%TOOLS%\caffe train --solver=.\ocr_recog_full_solver.prototxt

rem reduce learning rate by factor of 10
%TOOLS%\caffe train --solver=.\ocr_recog_full_solver_lr1.prototxt --snapshot=.\ocr_recog\ocr_recog_full_iter_60000.solverstate.h5

rem reduce learning rate by factor of 10
%TOOLS%\caffe train --solver=.\ocr_recog_full_solver_lr2.prototxt --snapshot=.\ocr_recog\ocr_recog_full_iter_65000.solverstate.h5
