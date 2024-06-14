import logging
from logging import handlers
import config.ocrdebug as ocrdebug

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日誌級別關係對映

    def __init__(self,filename,level='debug',when='H',backCount=168,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#設定日誌格式
        self.logger.setLevel(self.level_relations.get(level))#設定日誌級別
        sh = logging.StreamHandler()#往螢幕上輸出
        sh.setFormatter(format_str) #設定螢幕上顯示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往檔案裡寫入#指定間隔時間自動生成檔案的處理器
        #例項化TimedRotatingFileHandler
        #interval是時間間隔，backupCount是備份檔案的個數，如果超過這個個數，就會自動刪除，when是間隔的時間單位，單位有以下幾種：
        # S 秒
        # M 分
        # H 小時、
        # D 天、
        # W 每星期（interval==0時代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#設定檔案裡寫入的格式
        self.logger.addHandler(sh) #把物件加到logger裡
        self.logger.addHandler(th)


# import torch
# import time
# # from utils import get_file_list
# # from utils.ocr_util import cv2imread, rotate_image_up, resize_image, cut_boxes
# from utils.ocr_util import resize_image

# # strat build dbnmodel
# from post_processing import get_post_processing
# from data_loader import get_transforms
# from dbnmodels import build_model as build_dbnmodel
# # from utils.util import show_img, draw_bbox, save_result, get_file_list
# # import matplotlib.pyplot as plt
# # import pathlib

# class Dbn_model:
#     def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
#         '''
#         初始化pytorch模型
#         :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
#         :param gpu_id: 在哪一块gpu上运行
#         '''
#         self.gpu_id = gpu_id

#         if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
#             self.device = torch.device("cuda:%s" % self.gpu_id)
#         else:
#             self.device = torch.device("cpu")

#         if ocrdebug.ocrdebug:
#             # print('build Dbn_model device:', self.device)
#             pass        
#         checkpoint = torch.load(model_path, map_location=self.device)

#         dbn_config = checkpoint['config']
#         dbn_config['arch']['backbone']['pretrained'] = False
#         self.model = build_dbnmodel(dbn_config['arch'])
#         self.post_process = get_post_processing(dbn_config['post_processing'])
#         self.post_process.box_thresh = post_p_thre
#         self.img_mode = dbn_config['dataset']['train']['dataset']['args']['img_mode']
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.model.to(self.device)
#         self.model.eval()

#         self.transform = []
#         for t in dbn_config['dataset']['train']['dataset']['args']['transforms']:
#             if t['type'] in ['ToTensor', 'Normalize']:
#                 self.transform.append(t)
#         self.transform = get_transforms(self.transform)

#     # def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
#     def predict(self, imgarray: object, is_output_polygon=False, short_size: int = 1024):
#         '''
#         对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
#         :param img_path: 图像地址
#         :param is_numpy:
#         :return:
#         '''
#         # print('short_size:', short_size)
#         # assert os.path.exists(img_path), 'file is not exists'
#         # img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0) 
#         # img = cv2imread(img_path)
#         # img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
#         # if self.img_mode == 'RGB':
#         #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w = imgarray.shape[:2]
#         img = resize_image(imgarray, short_size)
#         # 将图片由(w,h)变为(1,img_channel,h,w)
#         tensor = self.transform(img)
#         tensor = tensor.unsqueeze_(0)

#         tensor = tensor.to(self.device)
#         batch = {'shape': [(h, w)]}
#         with torch.no_grad():
#             if str(self.device).__contains__('cuda'):
#                 torch.cuda.synchronize(self.device)
#             start = time.time()
#             preds = self.model(tensor)
#             if str(self.device).__contains__('cuda'):
#                 torch.cuda.synchronize(self.device)
#             box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
#             box_list, score_list = box_list[0], score_list[0]
#             if len(box_list) > 0:
#                 if is_output_polygon:
#                     idx = [x.sum() > 0 for x in box_list]
#                     box_list = [box_list[i] for i, v in enumerate(idx) if v]
#                     score_list = [score_list[i] for i, v in enumerate(idx) if v]
#                 else:
#                     idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
#                     box_list, score_list = box_list[idx], score_list[idx]
#             else:
#                 box_list, score_list = [], []
#             t = time.time() - start
#         return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t
    


