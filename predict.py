import socket
import time, tqdm, os
# import numpy as np
import anyconfig
import cv2
from PIL import Image
import torch
from utils.build import Logger


import argparse
def init_args():
    parser = argparse.ArgumentParser(description='HCH_OCR.pytorch')
    parser.add_argument('--drc_config_file', default='config/resnet18_FPN_Classhead.yaml', type=str, help='drcmodel config file')
    parser.add_argument('--drc_checkpoint_path', default=r'output\model.pth', type=str, help='drcmodel wehight path')
    parser.add_argument('--drc_flag', default='True', help='drcmodel work or skip, False is skip, True is work')
    parser.add_argument('--drc_input_folder', default='work/input', type=str, help='img folder path for inference')
    parser.add_argument('--drc_output_folder', default='work/output/post_img', type=str, help='img path for output')
    parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cuda or cpu(default)')  
    parser.add_argument('--ocrdebug', default='True', help='debug mode True or False(default)')
    parser.add_argument('--crnn_return', default='True', help='return result mode True or False(default)')
    parser.add_argument('--log_level', default='debug', help='log level debug or info(default)')
    args = parser.parse_args()
    return args

args = init_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
device = args.device
hostname = socket.gethostname()

import config.ocrdebug as ocrdebug
# print('ocrdebug.ocrdebug =', ocrdebug.ocrdebug)
if args.ocrdebug == 'True':
    ocrdebug.ocrdebug = True
    print('ocrdebug.ocrdebug =', ocrdebug.ocrdebug)

# start build drcmodel
drc_config = anyconfig.load(open(args.drc_config_file, 'rb'))
# from utils import strbase64to224array
from drcmodels import build_model as build_drcmodel
drc_config['arch']['backbone']['in_channels'] = 3
drcmodel = build_drcmodel(drc_config['arch'])
drc_checkpoint = torch.load(args.drc_checkpoint_path, map_location=device)
drcmodel.load_state_dict(drc_checkpoint)
# print('load drcmodel checkpoint')
drcmodel = drcmodel.to(device)
drcmodel.eval()


from utils.ocr_util import cv2imread, rotate_image_up
from utils.ocr_util import purge_debug_folder
from utils.util import get_file_list

import matplotlib.pyplot as plt
import pathlib


def img_padding(img):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.copyMakeBorder(img, 0, 0, 0, h-w, cv2.BORDER_CONSTANT, value=(128,128,128))   # top, bottom, left, right
    elif w > h:
        img = cv2.copyMakeBorder(img, 0, w-h, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
    return img


if __name__ == '__main__':
    log = Logger('work/log/pythonOCR.log',level=args.log_level)
    log.logger.info('HCH OCR server start on {}'.format(hostname))
    log.logger.info('pytorch device: {}'.format(device))
    log.logger.info('ocrdebug.ocrdebug: {}'.format(ocrdebug.ocrdebug))
    # log.logger.info('crnn_mode: {}'.format(args.crnn_mode))
    log.logger.debug('all args: {}'.format(args))
    # if args.crnn_mode != "inference":
    purge_debug_folder()

    try:
        tStart = time.time() # 計時開始
        count = 0
        for img_path in (get_file_list(args.drc_input_folder, p_postfix=['.jpg','.JPG'])):
            img_path = pathlib.Path(img_path)
            # log.logger.debug('now processing image: {}'.format(img_path))
            img = cv2imread(img_path)
            img = img_padding(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # imgrotate = img.copy()

            if args.drc_flag == 'True':
                # print('drc_falg is True, so run drcmodel.')
                img224 = cv2.resize(img, (224, 224))
                if ocrdebug.ocrdebug:
                    # im = Image.fromarray(img224)
                    im = Image.fromarray(img)
                    im.save('work/output/post_img/{}_224.jpg'.format(img_path.stem), quality=100, format='JPEG')
                # with torch.no_grad():
                img224 = torch.from_numpy(img224/255).permute(2, 0, 1).unsqueeze(0).float().to(device)
                # 做文件方向的預測與轉向處理
                drcmodel.eval()  
                drc_label  = drcmodel(img224).argmax(dim=1).cpu().numpy()[0]
                log.logger.debug('file: {}, label: {}'.format(img_path, drc_label))
                imgrotate = rotate_image_up(img, drc_label)
                if ocrdebug.ocrdebug:
                    im = Image.fromarray(imgrotate)
                    im.save('work/output/post_img/{}_drc.jpg'.format(img_path.stem), quality=100, format='JPEG')

            count += 1
        tEnd = time.time() # 計時結束
        log.logger.info('job is process end. 檔案筆數 {} , 執行時間 {:.2f} sec'.format(count, (tEnd - tStart)))
    except:
            log.logger.exception("Catch an exception.", exc_info=True)

