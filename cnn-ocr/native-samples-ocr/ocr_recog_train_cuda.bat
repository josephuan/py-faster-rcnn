
set TOOLS=..\..\bin_cuda

%TOOLS%\caffe train --solver=.\ocr_recog_solver.prototxt

rem reduce learning rate by factor of 10 after 8 epochs
%TOOLS%\caffe train --solver=.\ocr_recog_solver_lr1.prototxt --snapshot=.\ocr_recog\ocr_recog_iter_4000.solverstate.h5
