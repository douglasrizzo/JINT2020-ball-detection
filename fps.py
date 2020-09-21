import os
from dodo_detector.detection import TFObjectDetector
from os.path import join
from tqdm import tqdm
import logging
import platform
from glob import glob

jint_dir = 'JINT2020-ball-detection'


def make_logger():
   logger = logging.getLogger('fps_mobilenets')
   logger.setLevel(logging.INFO)

   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # create console handler and set level to debug
   ch = logging.StreamHandler()
   ch.setLevel(logging.INFO)
   ch.setFormatter(formatter)
   logger.addHandler(ch)

   return logger, formatter


if __name__ == "__main__":
   logger, formatter = make_logger()
   fh = logging.FileHandler(join(jint_dir, 'fps.log'), 'a')
   fh.setLevel(logging.INFO)
   fh.setFormatter(formatter)
   logger.addHandler(fh)

   label_map = join(jint_dir, 'data/data.pbtxt')
   paths_to_all_videos = glob(
       join(jint_dir, 'soccer_ball_dataset/test/videos/fisheye/ball/video1_*cut.webm'))
   # paths_to_all_models = glob(join(jint_dir, 'networks/mobilenets/exported/*'))
   paths_to_all_models = glob(join(jint_dir, 'networks/mobilenets/*'))

   logger.info('Script being executed on computer {}'.format(platform.node()))

   cuda_variable = os.environ[
       'CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'not set'

   logger.info('CUDA_VISIBLE_DEVICES: {}'.format(cuda_variable))

   handler_added = False
   pbar = tqdm(total=len(paths_to_all_models) * len(paths_to_all_videos))

   for path_to_model in paths_to_all_models:
      model_name = os.path.basename(path_to_model)
      saved_model = join(path_to_model, 'saved_model')
      detector = TFObjectDetector(saved_model, label_map)

      if not handler_added:
         logger.info('Adding the log handler to the detector')
         detector.add_logging_handler(fh)
         handler_added = True

      for path_to_video in paths_to_all_videos:
         pbar.update()
         logger.info('FPS for {} on video {}...'.format(model_name, path_to_video))
         detector.from_video(path_to_video)
