@echo off

if exist train_lmdb12 rd /q /s train_lmdb12

echo create train_lmdb12...
"caffe/convert_imageset.exe" "" 12/label-train.txt train_lmdb12 --backend=mtcnn --shuffle=true

echo done.
pause