@echo off

if exist train_lmdb48 rd /q /s train_lmdb48

echo create train_lmdb48...
"caffe/convert_imageset.exe" "" 48/label-train.txt train_lmdb48 --backend=mtcnn --shuffle=true

echo done.
pause