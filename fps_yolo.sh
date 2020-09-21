#!/bin/zsh
CFGS=(yolov3.cfg yolov3-tiny.cfg yolov4.cfg yolov4-tiny.cfg)
WEIGHTS=(yolov3_final.weights yolov3-tiny_final.weights yolov4_final.weights yolov4-tiny_final.weights)
JINTDIR=JINT2020-ball-detection
arrsize=${#CFGS[*]}

for ii in $(seq $arrsize); do
  CFG=${CFGS[$ii]}
  WEIGHT=${WEIGHTS[$ii]}
  OUTPUT=${OUTPUTS[$ii]}
  BASENAME="${CFG%.*}"
  LOGFILE="${BASENAME}_fps_cpu.log"
  for VIDEO in ${JINTDIR}/soccer_ball_dataset/test/videos/fisheye/ball/video1_*_10.webm; do
    echo ${CFG} ${VIDEO} ${1}
    echo ${CFG} ${VIDEO}
    # >> ${JINTDIR}/${LOGFILE}
    CUDA_VISIBLE_DEVICES=${1} ./darknet detector demo ${JINTDIR}/data/yolo/obj.data ${JINTDIR}/networks/yolo/configs/${CFG} ${JINTDIR}/networks/yolo/weights_trained/${WEIGHT} ${VIDEO} -dont_show -ext_output
    # >> ${JINTDIR}/${LOGFILE}
  done
done
