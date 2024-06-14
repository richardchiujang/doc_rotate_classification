import cv2
import numpy as np
import math
import re
import time, os
from PIL import Image
import config.ocrdebug as ocrdebug
import torch
from torch.autograd import Variable

def cv2imread(img_path): 
    '''
    處理 windows cv2 讀取中文檔名的問題
    import cv2
    '''
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # -1 cv2.IMREAD_UNCHANGED, cv2.IMREAD_IGNORE_ORIENTATION

# 中文问题
def cv2imwrite(filename, img):
    cv2.imencode('.jpg', img)[1].tofile(filename) 

def rotate_image_up(imgori, drc_label):
    '''
    rotate image to up
    '''
    if drc_label == 0:
        return imgori
    elif drc_label == 3:
        return cv2.rotate(imgori, cv2.ROTATE_90_CLOCKWISE)
    elif drc_label == 2:
        return cv2.rotate(imgori, cv2.ROTATE_180)
    elif drc_label == 1:
        return cv2.rotate(imgori, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return imgori

def strbase64to224array(image_str): 
    '''
    for flask api use transform base64 str (input image str) to origarr, img224arr
    '''
    from PIL import Image
    import base64 
    from io import BytesIO
    # import numpy as np
    # 64 # 直接輸入字串非讀檔，   輸出origarr, img64arr
    # _img_data = base64.decodestring(image_str)
    _img_data = base64.b64decode(image_str)
    # image = open('./work/output/{}.jpg'.format(str(time.time())), "wb")
    # image.write(_img_data)
    # image.close()
    img = Image.open(BytesIO(_img_data))
    img = img.convert('RGB')  # 一律轉成 channel=3 避免 exception 
    # if ocrdebug.ocrdebug:
    #     img.save('./work/output/{}.jpg'.format(str(time.time())), quality=100, format='JPEG')
    orig = img.copy()
    origarr = np.array(orig) # orig image to arr for output  
    img = img.resize((224,224), Image.ANTIALIAS)   # Image.LANCZOS
    # img.save('./work/output/{}.jpg'.format(str(time.time())), quality=100, format='JPEG')
    img224arr = np.array(img)  # resize img to arr for idclf  
    return origarr, img224arr

def resize_image(img, short_size):
    '''
    for dbnet use
    '''
    height, width, _ = img.shape
    if height <= short_size or width <= short_size:   # 不放大  放大會造成小圖裡面的字放大很多 richard 
        new_height = height
        new_width = width
    else:
        if height < width:
            new_height = short_size
            new_width = new_height / height * width
        else:
            new_width = short_size
            new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def cut_boxes(boxes, img, img_name):
    '''
    從輸入文件(圖片)剪下文字框，使用函數 cut_boxes(boxes, img, img_name)
    '''
    transformed_boxes = []
    vertical_box_long_side_extend = 10  # 长边扩大長度(較小);句子長度的邊界擴大像素
    horizon_box_long_side_extend = 10

    vertical_box_horizon_side_extend = 6  # 短边扩大長度(較大);字高或字寬的邊界擴大像素
    vertical_box_horizon_side_soft_extend = 3  # 短边扩大長度(較小);字高或字寬的邊界擴大像素(太小的字高或字寬)
    horizon_box_vertical_side_extend = 6
    horizon_box_vertical_side_soft_extend = 3

    box_directions = []
    src_out_pts = []
    for i, box in enumerate(boxes):
        # 创建原始图像的四个角点坐标（按照左上、右上、右下、左下的顺序）
        src_pts = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
        if ocrdebug.ocrdebug:
            # print('左上、右上、右下、左下',[(x,y) for x,y in src_pts])
            pass
        

        # 计算原始矩形框的水平邊和垂直邊的长度
        horizon_length = max(np.linalg.norm(src_pts[0][0] - src_pts[1][0]), np.linalg.norm(src_pts[3][0] - src_pts[2][0]))
        vertical_length = max(np.linalg.norm(src_pts[2][1] - src_pts[1][1]), np.linalg.norm(src_pts[3][1] - src_pts[1][1]))
        if ocrdebug.ocrdebug:
            # print('horizon_length, vertical_length', horizon_length, vertical_length)
            pass
        
        # box_direction = None
        box_direction = None
        # 判斷字高(寬度)
        if vertical_length/horizon_length > 2.:  # [垂直] 如果 H 大於 W 的 "2 倍"，則視為垂直長框(直式)
            box_direction = 'vertical'
            vertical_length_extend = vertical_box_long_side_extend  # 垂直框 垂直邊(H) 上下 加寬一點點
            if horizon_length >= 46:  # 大的文字框，水平邊(W)擴大比例較大
                horizon_length_extend = vertical_box_horizon_side_extend  # 垂直框 水平邊(W)擴大比例較大
            else:
                horizon_length_extend = vertical_box_horizon_side_soft_extend  # 垂直框 水平邊(W)擴大比例較小
        else:   # [水平](橫式)
            box_direction = 'horizontal'
            horizon_length_extend = horizon_box_long_side_extend  # 水平框 水平邊(H) 左右 加寬一點點
            if vertical_length >= 46:  # 如果垂直的高度(字高)很大
                vertical_length_extend = horizon_box_vertical_side_extend 
            else:  
                vertical_length_extend = horizon_box_vertical_side_soft_extend # 小字框 加小一點
        # long_edge = horizon_edge
        # short_edge = vertical_edge

        # 根据扩大百分比计算长边和短边的增加量
        horizon_increase = max(int(horizon_length_extend ), 1) # 增加量，最小为1
        vertical_increase = max(int(vertical_length_extend ), 1) # 增加量，最小为1        
        if ocrdebug.ocrdebug:
            # print('horizon_increase, vertical_increase', horizon_increase, vertical_increase)
            pass

        # 扩大原始矩形的四个角点坐标 （按照左上、右上、右下、左下的顺序）
        src_pts[0] = src_pts[0][0]-horizon_increase   , src_pts[0][1]-vertical_increase
        src_pts[1] = src_pts[1][0]+horizon_increase+2 , src_pts[1][1]-vertical_increase     # x+2 因為垂直框習慣框線貼字右邊
        src_pts[2] = src_pts[2][0]+horizon_increase+2 , src_pts[2][1]+vertical_increase+1 
        src_pts[3] = src_pts[3][0]-horizon_increase   , src_pts[3][1]+vertical_increase+1   # y+1 因為水平底線緊密，所以多加一點點 
        if ocrdebug.ocrdebug:
            # print('extened src_pts by increase length', [(x,y) for x,y in src_pts])
            pass

        # 边界处理，确保坐标不超过原图范围
        '''
        在这段代码中，np.clip() 函数用于限制坐标的范围，确保其不超过原图的宽度和高度范围。
        src_pts[:, 0] 表示取所有点的横坐标，src_pts[:, 1] 表示取所有点的纵坐标。
        通过 np.clip() 函数，将横坐标限制在 0 到 img_width - 1 的范围内，
        将纵坐标限制在 0 到 img_height - 1 的范围内。最后，打印修改后的坐标，以便进行检查。
        '''
        img_height, img_width = img.shape[:2]
        src_pts[:, 0] = np.clip(src_pts[:, 0], 0, img_width - 1)
        src_pts[:, 1] = np.clip(src_pts[:, 1], 0, img_height - 1)
        if ocrdebug.ocrdebug:
            # print('modify src_pts by np.clip()', [(x,y) for x,y in src_pts])
            pass

        # 计算目标图像的长宽
        target_width = horizon_length + 2 * horizon_increase+2
        target_height = vertical_length + 2 * vertical_increase+1
        if ocrdebug.ocrdebug:
            # print('target_width, target_height', target_width, target_height)
            pass

        # 创建目标图像的四个角点坐标
        dst_pts = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
        if ocrdebug.ocrdebug:
            # print('dst_pts', [(x,y) for x,y in dst_pts])
            pass

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 进行透视变换
        transformed_img = cv2.warpPerspective(img, M, (int(target_width), int(target_height)), flags=cv2.INTER_CUBIC)
        if ocrdebug.ocrdebug:
            # print('transformed_img.shape, order i of image', transformed_img.shape, i+1)
            pass

        transformed_boxes.append(transformed_img)

        # 保存转换后的图像
        if ocrdebug.ocrdebug:
            output_filename = r'work/output/cut_boxes/transformed_{}_{}.jpg'.format(img_name,i+1)
            im = Image.fromarray(transformed_img)
            im.save(output_filename, quality=100, format='JPEG')
            # cv2.imwrite(output_filename, transformed_img[:, :, ::-1])
        i += 1
        box_directions.append(box_direction)
        src_out_pts.append(src_pts)

    if ocrdebug.ocrdebug:
        output_filename = r'work/output/cut_boxes/ori_transformed_{}.jpg'.format(img_name)
        im = Image.fromarray(img)
        im.save(output_filename, quality=100, format='JPEG')
        # print('ocr_util.cut_box ori img type and shape', type(img), img.shape)
        # cv2.imwrite(output_filename, img[:, :, ::-1])
    src_out_pts = np.array(src_out_pts)
    return transformed_boxes, src_out_pts, box_directions


def rotate_cut_boxes(transformed_boxes, img_name):
    rotate_transformed_boxes = []
    for i, transformed_img in enumerate(transformed_boxes):
        # 判斷圖像是否為直式
        img_height, img_width = transformed_img.shape[:2]
        if img_height > img_width * 3:  # 直式圖像做旋轉處理 高度大於水平的3倍才視為直式，避免單一個字的圖片被旋轉
            # 進行旋轉
            rotated_img = cv2.rotate(transformed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotate_transformed_boxes.append(rotated_img)
            if ocrdebug.ocrdebug:
                output_filename = r'work/output/rotate_boxes/rotate_transformed_{}_{}.jpg'.format(img_name,i+1)
                im = Image.fromarray(rotated_img)
                im.save(output_filename, quality=100, format='JPEG')
                # cv2.imwrite(output_filename, rotated_img[:, :, ::-1])
        else:
            rotate_transformed_boxes.append(transformed_img)
    return rotate_transformed_boxes


def padding_boxes(rotate_transformed_boxes, img_name):
    '''
    將所有文字框轉成 H=32, W=max(w) 的圖片，高度等比例縮放，寬度補白邊，使用函數 padding_boxes()
    等比例縮放高度至 32
    寬度補白邊至 max(w)
    '''
    # 等比例縮放高度至 32
    max_width = 500
    same_height_boxes = []
    for i, rotated_img in enumerate(rotate_transformed_boxes):
        img_height, img_width = rotated_img.shape[:2]
        scale_ratio = 32 / img_height
        # 根据缩放比例调整图像的大小
        new_height = 32    # int(img_height * scale_ratio)  ### must fixed to 32  
        new_width = int(img_width * scale_ratio)
        rotated_img = cv2.resize(rotated_img, (new_width, new_height))
        same_height_boxes.append(rotated_img)
        img_height, img_width = rotated_img.shape[:2]
        if img_width > max_width:
            max_width = img_width

    # 將所有文字框補白邊至 max(w)
    for i, rotated_img in enumerate(same_height_boxes):
        img_height, img_width = rotated_img.shape[:2]
        if img_width < max_width:
            # 创建一个空白图像
            blank_image = np.ones((img_height, max_width, 3), np.uint8) * 255
            # 计算文字框的左上角坐标
            x_offset = int((max_width - img_width) / 2)
            # 将文字框放置在空白图像上
            blank_image[:, x_offset:x_offset + img_width] = rotated_img
            same_height_boxes[i] = blank_image
            if ocrdebug.ocrdebug:
                output_filename = r'work/output/padding_boxes/padding_{}_{}.jpg'.format(img_name,i+1)
                im = Image.fromarray(blank_image)
                im.save(output_filename, quality=100, format='JPEG')
                # cv2.imwrite(output_filename, blank_image[:, :, ::-1])
    # print('max_width', max_width)
    return same_height_boxes, max_width


def crnn_batch_recognition(config, imgs, model, converter, device, crnn_output_folder, width, boxes_list, box_directions, img_name, mode=None):
    model.eval()
    array_imgs = np.array(imgs)
    # gray_imgs = []
    gray_imgs = np.zeros((len(imgs), 1, 32, width), dtype=np.float32)
    for i in range(len(imgs)):
        gimg = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        gimg = np.reshape(gimg, (32, width, 1))
        gray_imgs[i] = gimg.transpose([2, 0, 1])
    # 将数据类型转换为 float32，并缩放到 0 到 1 的范围
    # normalized_imgs = (np.array(gray_imgs, dtype=np.float32) / 255.0 - config.DATASET.MEAN) / config.DATASET.STD
    # img = (img/255. - self.mean) / self.std
    normalized_imgs = (np.array(gray_imgs, dtype=np.float32) / 255.0  - config.DATASET.MEAN) / config.DATASET.STD   # array like (65, 1, 32, 834) (B, C, H, W)

    img = torch.from_numpy(normalized_imgs)
    inp = img.view(*img.size())   # img.view(1, *img.size())   # view 就是重構 tensor 的形狀，但是不會改變 tensor 的內容 如 torch.Size([65, 1, 32, 834])
    inp = inp.to(device)
    started = time.time()
    preds = model(inp)      # 經過 crnn model 後的結果 如 torch.Size([209, 65, 7924])  # [w, b, c]  [seq_len, batch, num_classes] [每條文字的長度, 文字總條數, 字典的長度]

    # 設定閾值 小於 閾值 的都設定成 -10
    preds[preds <= -0.5000] = -10

    # out = preds[:, 6, :]
    # np.savetxt('C:\\develop\\hch_ocr\\work\\output\\ocr_result\\{}.preds.txt'.format(img_name), out.detach().numpy())
    # print('preds', preds.shape, type(preds))   # torch.Size([209, 65, 7924]) <class 'torch.Tensor'>
    finished = time.time()
    sim_preds_time = time.time()
    batch_size = inp.size(0)   # 65 
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)    # create a int tensor [209*65] = 13585,  torch.Size([65]) 
    _, preds = preds.max(2)   # torch.Size([209, 65])
    # np.savetxt('C:\\develop\\hch_ocr\\work\\output\\ocr_result\\{}.max.txt'.format(img_name), preds.detach().numpy())
    preds = preds.transpose(1, 0).contiguous().view(-1)  # torch.Size([13585])   # contiguous() 會讓 tensor 在記憶體中連續
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    if type(sim_preds) != list: # 如果只有一個結果，轉成 list，否則在後面的 for 迴圈 enumerate(sim_preds)會把字串拆成一個一個字，最後就剩下第一個字留下而已
        sim_preds = [sim_preds]
    
    ocr_results = []
    for i, sim_pred in enumerate(sim_preds):
        if ocrdebug.ocrdebug:
            # print('{} {} {}'.format(img_name, [(x,y) for x,y in boxes_list[i]], sim_pred))
            pass
        # ocr_results.append('result {}:{} {} {} '.format(i+1, sim_pred, [(x,y) for x,y in boxes_list[i]], img_name))
        ocr_results.append('{:03d} {}'.format(i+1, sim_pred))

    data=[]
    entries = []
    for ocr_result, box_direction, bbox in zip(ocr_results, box_directions, boxes_list):
        if box_direction == 'vertical':
            box_direction = 'V'
        else:
            box_direction = 'H'
        d = '{} {} {}\n'.format(ocr_result, box_direction, [(x,y) for x,y in bbox])
        data.append(d)

    # 將每個條目解析並將其存入一個字典中
    for line in data:
        match = re.match(r'(\d{3}) (.*?) ([HV]) (\[.*?\])', line)
        if match:
            seq, text, direction, coords = match.groups()
            coords = eval(coords)
            # print(seq, text, direction, coords, coords[0], coords[1])
            entries.append({
                'seq': seq,
                'text': text,
                'direction': direction,
                'coords': coords
            })


    # 將 H 與 V 分開
    entries_h = [entry for entry in entries if entry['direction'] == 'H']
    entries_v = [entry for entry in entries if entry['direction'] == 'V']

    # 分別進行排序
    entries_h.sort(key=lambda e: e['coords'][0][0]**2 + e['coords'][0][1]**2) # 左上到右下
    entries_v.sort(key=lambda e: e['coords'][1][0]**2 + e['coords'][1][1]**2, reverse=True) # 右上到左下 , reverse=True

    # 合併排序後的資料
    sorted_data = entries_h + entries_v

    # 將排序後的資料轉為字串
    pure_data = [f"{e['text']}" for e in sorted_data] 
    sorted_data = [f"{e['seq']} {e['text']} {e['direction']} {e['coords']}" for e in sorted_data]    

    
    

    if ocrdebug.ocrdebug:
        # print('sim_preds time: {0}'.format(time.time() - sim_preds_time))
        # print('preds time: {0}'.format(finished - started))
        pass

    # if ocrdebug.ocrdebug:
    # output_filename = r'work/output/ocr_result/{}.txt'.format(img_name)
    if mode == 'inference':
        output_folder = r'{}'.format(crnn_output_folder)
        isExist = os.path.exists(output_folder)
        if not isExist:
            os.makedirs(output_folder)
        output_filename = r'{}/{}.txt'.format(crnn_output_folder, img_name)
    else:
        output_filename = r'work/output/ocr_result/{}.txt'.format(img_name)
    with open(output_filename, 'w', encoding='utf-8') as f:
        for res in pure_data:
            f.write('{}\n'.format(res))
        # for i, result in enumerate(sorted_data):
        #     f.write('{}\n'.format(result[4:]))

    f.close()
    return pure_data

        # for i, (ocr_result, box_direction, bbox) in enumerate(zip(ocr_results, box_directions, boxes_list)):
        #     if box_direction == 'vertical':
        #         box_direction = 'V'
        #     else:
        #         box_direction = 'H'
        #     f.write('{} {} {}\n'.format(ocr_result, box_direction, [(x,y) for x,y in bbox]))


def combine_box(boxes_list):
    new_boxes_list = []
    for i in range(len(boxes_list)): # 這是原來的文字框
        current_box = boxes_list[i]
        current_box_angle = calculate_angle(current_box) # 計算文字框的角度

        if not is_horizontal_box(current_box_angle) and not is_vertical_box(current_box_angle): # 如果是傾斜的文字框，不合併
            new_boxes_list.append(current_box)
            continue # 跳過這一次，繼續迴圈

        current_box_direction = is_horizon_or_vertical_box(current_box) # 判斷文字框是水平還是垂直
        merged = False

        for j in range(len(new_boxes_list)): # 這是合併後的文字框
            new_box = new_boxes_list[j]
            new_box_angle = calculate_angle(new_box) # 計算文字框的角度

            # 如果是傾斜的文字框，不合併
            if not is_horizontal_box(new_box_angle) and not is_vertical_box(new_box_angle):
                continue

            new_box_direction = is_horizon_or_vertical_box(new_box) # 判斷文字框是水平還是垂直

            if current_box_direction == new_box_direction: # 如果文字框方向一樣，才能合併
                
                if current_box_direction == 'horizontal': # 如果文字框是水平的
                    overlap = horizon_box_overlap(current_box, new_box) # 判斷文字框是否重疊
                else: # 如果文字框是垂直的
                    overlap = vertical_box_overlap(current_box, new_box) # 判斷文字框是否重疊

                if overlap:
                    merged = True
                    merged_box = merge_boxes(current_box, new_box)
                    new_boxes_list[j] = merged_box  # 將合併後的文字框放回合併後的文字框列表(取代原來的文字框)
                    # 這裡繼續合併迴圈，剩下的文字框還有可能和合併後的文字框重疊

        if not merged:
            new_boxes_list.append(current_box)

    return new_boxes_list


def calculate_angle(box):
    p1, p2, p3, p4 = box
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle


def is_horizon_or_vertical_box(box):
    p1, p2, p3, p4 = box
    # 計算長邊和短邊的長度
    width_side = np.linalg.norm(p1 - p2)
    high_side = np.linalg.norm(p2 - p3)
    if width_side * 3 < high_side: # 高度大於水平3倍，視為垂直長框(直式)
        return 'vertical'
    else:
        return 'horizontal'


def is_horizontal_box(angle):
    return abs(angle) < 15


def is_vertical_box(angle):
    return 75 <= abs(angle) <= 105


def horizon_box_overlap(box1, box2):
    x1_box1, y1_box1 = box1[0]
    x2_box1, y2_box1 = box1[1]
    x3_box1, y3_box1 = box1[2]
    x4_box1, y4_box1 = box1[3]
    xmin_box1 = min(x1_box1, x4_box1, x2_box1, x3_box1)
    xmax_box1 = max(x1_box1, x4_box1, x2_box1, x3_box1)
    ymin_box1 = min(y1_box1, y4_box1, y2_box1, y3_box1)
    ymax_box1 = max(y1_box1, y4_box1, y2_box1, y3_box1)
    x1_box2, y1_box2 = box2[0]
    x2_box2, y2_box2 = box2[1]
    x3_box2, y3_box2 = box2[2]
    x4_box2, y4_box2 = box2[3]
    xmin_box2 = min(x1_box2, x4_box2, x2_box2, x3_box2)
    xmax_box2 = max(x1_box2, x4_box2, x2_box2, x3_box2)
    ymin_box2 = min(y1_box2, y4_box2, y2_box2, y3_box2)
    ymax_box2 = max(y1_box2, y4_box2, y2_box2, y3_box2)

    if xmin_box1 <= xmin_box2: # box1 在 box2 的左邊
        if xmax_box1 + 5 >= xmin_box2 - 5: # box1 的右邊有重疊 box2 的左邊(-3)
            if ymin_box1 <= ymin_box2 <= ymax_box1 or ymin_box1 <= ymax_box2 <= ymax_box1: # box2 的上下有重疊 box1 的上下
                # 垂直部分重疊超過 50% 才視為重疊
                if abs(max(ymin_box1, ymin_box2) - min(ymax_box1, ymax_box2)) / max((ymax_box1-ymin_box1), (ymax_box2-ymin_box2)) >= 0.5:
                    return True
    else: # box1 在 box2 的右邊
        if xmin_box1 - 5 <= xmax_box2 + 5: # box1 的左邊有重疊 box2 的右邊(+3)
            if ymin_box1 <= ymin_box2 <= ymax_box1 or ymin_box1 <= ymax_box2 <= ymax_box1: # box2 的上下有重疊 box1 的上下
                # 垂直部分重疊超過 50% 才視為重疊
                if abs(max(ymin_box1, ymin_box2) - min(ymax_box1, ymax_box2)) / max((ymax_box1-ymin_box1), (ymax_box2-ymin_box2)) >= 0.5:
                    return True
    return False


def vertical_box_overlap(box1, box2):
    x1_box1, y1_box1 = box1[0]
    x2_box1, y2_box1 = box1[1]
    x3_box1, y3_box1 = box1[2]
    x4_box1, y4_box1 = box1[3]
    xmin_box1 = min(x1_box1, x4_box1, x2_box1, x3_box1)
    xmax_box1 = max(x1_box1, x4_box1, x2_box1, x3_box1)
    ymin_box1 = min(y1_box1, y4_box1, y2_box1, y3_box1)
    ymax_box1 = max(y1_box1, y4_box1, y2_box1, y3_box1)
    x1_box2, y1_box2 = box2[0]
    x2_box2, y2_box2 = box2[1]
    x3_box2, y3_box2 = box2[2]
    x4_box2, y4_box2 = box2[3]
    xmin_box2 = min(x1_box2, x4_box2, x2_box2, x3_box2)
    xmax_box2 = max(x1_box2, x4_box2, x2_box2, x3_box2)
    ymin_box2 = min(y1_box2, y4_box2, y2_box2, y3_box2)
    ymax_box2 = max(y1_box2, y4_box2, y2_box2, y3_box2)

    if ymin_box1 <= ymin_box2: # box1 在 box2 的上面
        if ymax_box1 + 3 >= ymin_box2 - 3: # box1 的下面有重疊 box2 的上面(-3)
            if xmin_box1 <= xmin_box2 <= xmax_box1 or xmin_box1 <= xmax_box2 <= xmax_box1: # box2 的左右有重疊 box1 的左右
                # 水平部分重疊超過 50% 才視為重疊
                if abs(max(xmin_box1, xmin_box2) - min(xmax_box1, xmax_box2)) / max((xmax_box1-xmin_box1), (xmax_box2-xmin_box2)) >= 0.5:
                    return True
    else: # box1 在 box2 的下面
        if ymin_box1 - 3 <= ymax_box2 + 3: # box1 的上面有重疊 box2 的下面(+3)
            if xmin_box1 <= xmin_box2 <= xmax_box1 or xmin_box1 <= xmax_box2 <= xmax_box1: # box2 的左右有重疊 box1 的左右
                # 水平部分重疊超過 50% 才視為重疊
                if abs(max(xmin_box1, xmin_box2) - min(xmax_box1, xmax_box2)) / max((xmax_box1-xmin_box1), (xmax_box2-xmin_box2)) >= 0.5:
                    return True
    return False

def merge_boxes(box1, box2):
    x1_box1, y1_box1 = box1[0]
    x2_box1, y2_box1 = box1[1]
    x3_box1, y3_box1 = box1[2]
    x4_box1, y4_box1 = box1[3]
    xmin_box1 = min(x1_box1, x4_box1, x2_box1, x3_box1)
    xmax_box1 = max(x1_box1, x4_box1, x2_box1, x3_box1)
    ymin_box1 = min(y1_box1, y4_box1, y2_box1, y3_box1)
    ymax_box1 = max(y1_box1, y4_box1, y2_box1, y3_box1)
    x1_box2, y1_box2 = box2[0]
    x2_box2, y2_box2 = box2[1]
    x3_box2, y3_box2 = box2[2]
    x4_box2, y4_box2 = box2[3]
    xmin_box2 = min(x1_box2, x4_box2, x2_box2, x3_box2)
    xmax_box2 = max(x1_box2, x4_box2, x2_box2, x3_box2)
    ymin_box2 = min(y1_box2, y4_box2, y2_box2, y3_box2)
    ymax_box2 = max(y1_box2, y4_box2, y2_box2, y3_box2)

    xmin = min(xmin_box1, xmin_box2)
    xmax = max(xmax_box1, xmax_box2)
    ymin = min(ymin_box1, ymin_box2)
    ymax = max(ymax_box1, ymax_box2)

    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)


def sort_boxes(boxes):
    # 自定义排序关键字函数
    def sort_key(box):
        return box[0, 0] + box[0, 1]   # 框的左上角坐标之和作为排序依据
    
    # 将boxes转换为NumPy数组
    boxes = np.array(boxes)
    
    # 使用自定义排序关键字函数对框进行排序
    sorted_boxes = sorted(boxes, key=sort_key)
    
    # 将排序后的框转换为NumPy数组并返回
    return np.array(sorted_boxes)


def purge_debug_folder():
    '''
    purge and mkdir
    ['cut_boxes', 'rotate_boxes', 'padding_boxes', 'post_img', 'inference', 'ocr_result']
    '''
    from shutil import rmtree

    if os.path.exists('work/output/cut_boxes'):
        rmtree('work/output/cut_boxes')
    if os.path.exists('work/output/rotate_boxes'):
        rmtree('work/output/rotate_boxes')
    if os.path.exists('work/output/padding_boxes'):
        rmtree('work/output/padding_boxes')
    if os.path.exists('work/output/ocr_result'):
        rmtree('work/output/ocr_result')
    if os.path.exists('work/output/post_img'):
        rmtree('work/output/post_img')
    if os.path.exists('work/output/inference'):
        rmtree('work/output/inference')

    os.makedirs(('work/output/cut_boxes'), exist_ok=True)
    os.makedirs(('work/output/rotate_boxes'), exist_ok=True)
    os.makedirs(('work/output/padding_boxes'), exist_ok=True)
    os.makedirs(('work/output/ocr_result'), exist_ok=True)
    os.makedirs(('work/output/post_img'), exist_ok=True)
    os.makedirs(('work/output/inference'), exist_ok=True)

    time.sleep(1)

    print('purge and mkdir work/output/cut_boxes, rotate_boxes, padding_boxes, post_img, inference, ocr_result')



