# Data from the paper "Detecting soccer balls with reduced neural networks"

This directory contains the results from our JINT 2020 paper, in which we trained multiple MobileNetV2 and V3 as well as YOLO and TinyYOLO v3 and v4 models on a soccer ball image dataset collected from humanoid robots.

<!-- TOC -->

- [Data from the paper "Detecting soccer balls with reduced neural networks"](#data-from-the-paper-detecting-soccer-balls-with-reduced-neural-networks)
  - [Dataset](#dataset)
  - [Networks](#networks)
  - [Other software](#other-software)
  - [Training the networks](#training-the-networks)
    - [Creating the necessary files](#creating-the-necessary-files)
      - [For YOLO](#for-yolo)
      - [For MobileNets](#for-mobilenets)
    - [Training YOLO](#training-yolo)
    - [Training MobileNets](#training-mobilenets)
  - [Getting mAP and FPS](#getting-map-and-fps)

<!-- /TOC -->

## Dataset

This is the data set we used:

> R. A. C. Bianchi, “Open soccer ball dataset,” 2020, doi: 10/ghcfxn.

It is available for free on [IEEE Dataport](https://ieee-dataport.org/open-access/open-soccer-ball-dataset) as long as you have a free account.

## Networks

To train MobileNets, we used the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/), on TensorFlow 1.15, but 2.2 onwards is also compatible.

The pretrained models were taken from the [slim classification models page]((https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md)). The config pipelines to train each one of them, just like we did, are available in `networks/mobilenets`. The trained networks are available in the same directory.

YOLO was taken from the new [official darknet repository](https://github.com/AlexeyAB/darknet). For training, the `Makefile` was changed so the project is compiled for the V100 GPU. Compilation was then done with the command `make GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=1 AVX=1 OPENMP=1`. The tutorial used to configure and run the training was on [the README](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).

The trained weights and config files are available in `networks/yolo`.

## Other software

- [labelImg](https://github.com/tzutalin/labelImg): to label the images;
- [dodo_detector](https://github.com/douglasrizzo/dodo_detector): to load the trained MobileNets, apply them to videos and capture the inference times;
- [detection_util_scripts](https://github.com/douglasrizzo/dodo_detector): to generate files necessary to train the MobileNets and YOLO models.

## Training the networks

This is a simplified tutorial on how to generate the necessary files to train the neural networks. I'll assume:

this repository was downloaded in a directory called `JINT2020-ball-detection`;
the Open Soccer Ball Dataset was downloaded to a directory called `soccer_ball_dataset`;
the detection_util_scripts repository was downloaded to a directory in `~/detection_util_scripts`.

### Creating the necessary files

If the CSV files are missing (they are needed to generate other possibly missing files):

```sh
python ~/detection_util_scripts/generate_csv.py xml soccer_ball_dataset/dataset/training/annotations JINT2020-ball-detection/data/train.csv
python ~/detection_util_scripts/generate_csv.py xml soccer_ball_dataset/dataset/test/ball/annotations JINT2020-ball-detection/data/eval.csv
```

#### For YOLO

If the YOLO `obj.names` file is missing:

```sh
echo 'ball ' > JINT2020-ball-detection/data/yolo/obj.names
```

If the `.txt` annotation files for YOLO are missing:

```sh
python ~/detection_util_scripts/generate_yolo_txt.py JINT2020-ball-detection/data/train.csv JINT2020-ball-detection/data/yolo/obj.names JINT2020-ball-detection/data/yolo/train
python ~/detection_util_scripts/generate_yolo_txt.py JINT2020-ball-detection/data/eval.csv JINT2020-ball-detection/data/yolo/obj.names JINT2020-ball-detection/data/yolo/eval
```

If the lists of images for YOLO are missing:

```sh
find soccer_ball_dataset/training/images -type f -name '*.jpg' > JINT2020-ball-detection/data/yolo/train.txt
find soccer_ball_dataset/test/ball/img -type f -name '*.jpg' > JINT2020-ball-detection/data/yolo/test.txt
```

#### For MobileNets

If the label map is missing:

```sh
python ~/detection_util_scripts/generate_pbtxt.py csv JINT2020-ball-detection/data/train.csv JINT2020-ball-detection/data/mobilenets/data.pbtxt
```

If the TFRecords are missing:

```sh
python ~/detection_util_scripts/generate_tfrecord.py JINT2020-ball-detection/data/train.csv JINT2020-ball-detection/data/data.pbtxt soccer_ball_dataset/training/images train.record

python ~/detection_util_scripts/generate_tfrecord.py JINT2020-ball-detection/data/eval.csv JINT2020-ball-detection/data/data.pbtxt soccer_ball_dataset/test/ball/img eval.record
```

### Training YOLO

I assume you have already downloaded the darknet repository and compiled it.

1. Create symbolic links from all image files in the soccer ball data set to the directories where YOLO annotations are kept.

  ```sh
  ln -s soccer-ball-dataset/training/images/*.jpg JINT2020-ball-detection/data/yolo/train/
  ln -s soccer-ball-dataset/test/ball/img/*.jpg JINT2020-ball-detection/data/yolo/eval/
  ```

2. Download the pretrained weights. These are direct links for YOLOv3/v4 [[1]](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) [[2]](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp). If they don't work anymore. Please go to [the official repository](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects ) and find up-to-date links there. The same goes for TinyYOLO [[link]](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29).

3. From the darknet repo root directory, run something like the command below. Please note that the last argument must point to the weights downloaded in the previous step:

  ```sh
  ./darknet detector train JINT2020-ball-detection/data/yolo/obj.data JINT2020-ball-detection/networks/yolo/configs/yolov4.cfg JINT2020-ball-detection/networks/yolo/weights_pretrained/yolov4.weights
  ```

  Select different `.cfg` files and the corresponding weights files depending on the network you want to train.

  New weights should be present in `networks/yolo/weights_trained`.

### Training MobileNets

To train a MobileNet:

- download and install TensorFlow;
- download the [TensorFlow Model Garden](https://github.com/tensorflow/models) and follow the installation instructions in `models/research/object_detection`;
- download one of the MobileNet models from the [slim classification models page]((https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md));
- use the corresponding training pipeline provided in this repository, in the `networks/mobilenets` directory.

Run the following command from the `models/research` directory of the `models` repository:

```sh
python object_detection/model_main.py \
  --pipeline_config_path=${PIPELINEPATH} \
  --model_dir=${MODELDIR} \
  --num_train_steps=50000 \
  --sample_1_of_n_eval_examples=1 \
  --alsologtostderr
```

where `PIPELINEPATH` is the path to one of the `pipeline.config` files present in one of the subdirectories in `networks/mobilenets` and `MODELDIR` is the path to a directory that will keep checkpoint files.

## Getting mAP and FPS

**mAP for YOLO:**

Example:

```sh
./darknet detector map JINT2020-ball-detection/data/yolo/obj.data JINT2020-ball-detection/networks/yolo/configs/yolov3.cfg JINT2020-ball-detection/networks/yolo/weights_trained/yolov3_final.weights -points 0
```

**FPS for YOLO:**

There is a `fps_yolo.sh` script in the root folder, which should be run from the darknet root folder. It will load the `.cfg` and `.weights` files, apply each network from the `networks/yolo/configs` and `networks/yolo/weights_trained` directories to the videos in `dataset/test/videos/fisheye/ball` and save individual log files for each network in the root folder of this project. I recommend you open `fps_yolo.sh`, as there are some hardcoded paths inside the script.

**mAP for MobileNets:** after training the networks, we got the contents of the mAP@.50IOU graph on TensorBoard.

**FPS for MobileNets:**

```sh
# install an object detection package that actually calculates the FPS
pip install git+https://github.com/douglasrizzo/dodo_detector.git@0.7.1
# measures on CPU
CUDA_VISIBLE_DEVICES=-1 python fps.py
# measures on GPU
CUDA_VISIBLE_DEVICES=0 python fps.py
```

Then check `fps.log`.
