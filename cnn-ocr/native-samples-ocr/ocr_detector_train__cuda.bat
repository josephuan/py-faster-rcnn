
set TOOLS=..\..\bin_cuda

%TOOLS%\caffe train --solver=.\ocr_detector_solver.prototxt

rem reduce learning rate by factor of 10 after 8 epochs
%TOOLS%\caffe train --solver=.\ocr_detector_solver_lr1.prototxt --snapshot=.\ocr_detector\ocr_detector_iter_4000.solverstate.h5
