# TODO: create shell script for running your YoloV1-vgg16bn model
#!/bin/bash

{
    wget "https://www.dropbox.com/s/y1hpstbi9cjcuek/yolo_80_model.pth?dl=1"
	python ./testing.py $1 $2
}||{
    wget "https://www.dropbox.com/s/y1hpstbi9cjcuek/yolo_80_model.pth?dl=1"
	python3 ./testing.py $1 $2
}
