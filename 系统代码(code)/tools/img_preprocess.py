import cv2
import numpy as np
from skimage.morphology import skeletonize
from config import IMG_SIZE,FILELIST,MODEL_DIR,\
    SPACIAL_RELATIONSHIP as spartial_relationship,\
    LARGEST_NUMBER_OF_SYMBOLS,SCALSIZE
from matplotlib import pyplot as plot


def read_img_and_convert_to_binary(filename):
    '''讀取圖片並將圖片做前處理, 返回原彩色圖和黑白(0/1)圖, 大小倍率透過 config.SCALSIZE 進行調整
    包含去雜訊、灰階、黑白化(0與1)

    Args:
        filename (str): 絕對 / 相對圖片完整路徑 ex: './testImgs/easy +/3.jpg'

    Attributes:
        config.SCALSIZE (number): 圖片倍率

    Return:
        original_img (array): 原圖 (b,g,r)
        binary_img (array): 黑白圖
    '''
    #读取待处理的图片
    original_img = cv2.imread(filename)
    # print(original_img)
    #将原图分辨率缩小SCALSIZE倍，减少计算复杂度
    original_img = cv2.resize(original_img,(np.int(original_img.shape[1]/SCALSIZE),np.int(original_img.shape[0]/SCALSIZE)), interpolation=cv2.INTER_AREA)
    #降噪
    blur = cv2.GaussianBlur(original_img, (5, 5), 0)
    #将彩色图转化成灰度图
    img_gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    #图片开（opening）处理，用来降噪，使图片中的字符边界更圆滑，没有皱褶
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

    kernel2 = np.ones((3,3), np.uint8)
    opening = cv2.dilate(opening, kernel2, iterations=1)
    # Otsu's thresholding after Gaussian filtering
    # 采用otsu阈值法将灰度图转化成只有0和1的黑白图
    blur = cv2.GaussianBlur(opening,(13,13),0)
    #ret, binary_img = cv2.threshold(img_gray, 120, 1, cv2.THRESH_BINARY_INV)
    ret,binary_img = cv2.threshold(blur,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return original_img,binary_img

def extract_img(location,img,contour=None):
    '''從img截取location區域的圖像，並縮放大小成IMG_SIZE*IMG_SIZE
    
    Args:
        location (tuple): 字符位置(x,y,寬,高) 
        img (array): 黑白圖片
        contour (array): 字符輪廓座標 , 如果為None 則單純裁切位置, 反之會過濾輪廓以外的雜訊
    Return:
        extracted_img (array): 已經縮放IMG_SIZE*IMG_SIZE的字符圖
    '''
    x,y,w,h=location
    # 只提取轮廓内的字符
    if contour is None:
        extracted_img = img[y:y + h, x:x + w]
    else:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        img_after_masked = cv2.bitwise_and(mask, img)
        extracted_img = img_after_masked[y:y + h, x:x + w]
    # 将提取出的img归一化成IMG_SIZE*IMG_SIZE大小的黑白图
    black = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    if (w > h):
        res = cv2.resize(extracted_img, (IMG_SIZE, (int)(h * IMG_SIZE / w)), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        black[d:res.shape[0] + d, 0:res.shape[1]] = res
    else:
        res = cv2.resize(extracted_img, ((int)(w * IMG_SIZE / h), IMG_SIZE), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        black[0:res.shape[0], d:res.shape[1] + d] = res
    extracted_img = skeletonize(black)
    extracted_img = np.logical_not(extracted_img)
    return extracted_img

def binary_img_segment(binary_img,original_img=None):
    '''將黑白圖裡面的字符切割成單個字符，返回三維數組，每一個字符是一個dict，包含字符所在位置大小location，以及字符切割黑白圖src_img ()
    Args:
        binary_img (array): 黑白圖
        original_img (array): 原始圖片
    Retunn:
        symbols (list): 每個經縮放後的字符圖
            字符格式為:{'location': x, y , 原始寬, 原始高,'src_img': array([...])}
            例如:
                [{'location': (39, 40, 17, 100), 'src_img': array([[ True,  True...,  True]])},
                {'location': (83, 54, 75, 86), 'src_img': array([[ True,  True...,  True]])}, ...]
    '''
    # binary_img = skeletonize(binary_img)
    # plot.imshow( binary_img,cmap = 'gray', interpolation = 'bicubic')
    # plot.show()
    #寻找每一个字符的轮廓，使用cv2.RETR_EXTERNAL模式，表示只需要每一个字符最外面的轮廓
    img, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_TREE
    #cv2.drawContours(img_original, contours, -1, (0, 255, 0), 2)
    if len(contours) > LARGEST_NUMBER_OF_SYMBOLS:
        raise ValueError('symtem cannot interpret this image!')
    symbol_segment_location = []
    # 将每一个联通体，作为一个字符
    symbol_segment_list = []
    index = 1
    for contour in contours:
        location = cv2.boundingRect(contour)
        x, y, w, h = location
        if(w*h<100):
            continue
        symbol_segment_location.append(location)
        # 只提取轮廓内的字符
        extracted_img = extract_img(location,img,contour)
        symbol_segment_list.append(extracted_img)
        if len(original_img):
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        symbols=[]
        for i in range(len(symbol_segment_location)):
            symbols.append({'location':symbol_segment_location[i],'src_img':symbol_segment_list[i]})
        # 对字符按字符横坐标排序
        symbols.sort(key=lambda x:x['location'][0])
    return symbols


