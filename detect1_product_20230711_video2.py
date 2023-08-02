import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
#from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *
from utils.cvfpscalc import CvFpsCalc
import collections
import numpy as np
import time

import threading
#from multiprocessing import Queue
from queue import Queue
class product:
    def __init__(self,same_num) :
        self.trend=[[] for _ in range(same_num)]#在list中創n個list空間(n=2) , 2代表每一類會有兩個物品
        self.history=[[] for _ in range(same_num)]
        self.aver_trend=[[] for _ in range(same_num)]
        self.exist=[[] for _ in range(same_num)]
        self.trend_analysis=[[] for _ in range(same_num)]
        self.loc_val=[[] for _ in range(same_num)]
        self.exist_val=[[] for _ in range(same_num)]
        self.exist_aver=[[] for _ in range(same_num)]
        self.cls_flag=0
        self.count=0
        self.flag_product=[[1] for _ in range(same_num)]

        
class vStream:
    def __init__(self,src,width,height):
        self.width=width
        self.height=height
        self.capture=cv2.VideoCapture(0)
        self.capture1=cv2.VideoCapture(1)

       
        _,self.frame = self.capture.read()
        _,self.frame1 = self.capture1.read()

        self.stop_eventvs = threading.Event() 
        self.thread=threading.Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
        print("success running...")
    def update(self):
        global  thread_flag
        if not self.capture.isOpened() and  not self.capture1.isOpened():
            print("...can not open video...")
        while thread_flag:

            self.capture.grab()
            _,self.frame=self.capture.retrieve()
            #_,self.frame = self.capture.read()
            self.capture1.grab()
            _,self.frame1=self.capture1.retrieve()

            #self.frame2=cv2.resize(self.frame,(self.width,self.height))
            time.sleep(0.01)  # wait time
    def __iter__(self):#one time
        #self.capture.isOpened()
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        #return self.frame
        return self.frame, self.frame1
    def stop(self):
        self.capture.release()
        #self.thread.join() 
        self.stop_eventvs.set()
        
        
    

def moving_aver_exist(product_exist,exist,n):
    product_exist.append(exist)
    if len(product_exist)>n:
    	product_exist.pop(0)
    return sum(product_exist)/n
'''
def moving_aver_exist2(product_exist2,exist2,n):
    product_exist2.append(exist2)
    if len(product_exist2)>n:
    	product_exist2.pop(0)
    return sum(product_exist2)/n
'''
def trend_yaxis(point_trend,point1,point2):
    #print('point1',point1,'point2',point2)

    if point1[1]-point2[1]>2:
    	point_trend.append(1);
    elif point2[1]-point1[1]>2:
    	point_trend.append(-1);
    else:
    	point_trend.append(0);	
    

    return point_trend
#------------------------------------------------
def moving_aver_trend(product_aver_trend,trend,n):
    product_aver_trend.append(trend)
    if len(product_aver_trend)>n:
    	product_aver_trend.pop(0)
    return sum(product_aver_trend)/n

def moving_aver_trend_analysis2(product_trend_analysis,product_trend):
    move_mark=[]
    plus_flag=0
    minus_flag=0
    product_count=0
    flag=0
    flag2=0
    for i in product_trend:
        var_move=moving_aver_trend(product_trend_analysis,i,5)
        #print("product_trend_analysis",product_trend_analysis)
        #print("var_move",var_move)
        if var_move>=0.6 and plus_flag==0:
            move_mark.append("plus")
            plus_flag=1
        elif var_move<=-0.6 and minus_flag==0:
            move_mark.append("minus")
            minus_flag=1

    #print("move_mark",move_mark)

    if len(move_mark)>=2:
        if move_mark[0]=="plus" and move_mark[1]=="minus":
            product_count=0
        if move_mark[0]=="minus" and move_mark[1]=="plus":
            product_count=0
    elif len(move_mark)==1:
        if move_mark[0]=="plus":
            product_count=-1
        if move_mark[0]=="minus":
            product_count=1

    return product_count

#------------------------------------------------
def product_aver_direction(product_trend,product_aver_trend,product_history):

    trend_yaxis(product_trend,product_history[-1],product_history[-2])
    #print("product_trend",product_trend)
    #print("point_trend",point_trend)
    move_trend=moving_aver_trend(product_aver_trend,product_trend[-1],5)
    if move_trend>=0.3:
        product_loc="goin"
    elif move_trend<=-0.3:
        product_loc="back"
    else:
        product_loc="stop"	

    #if len(product_trend)>=10:
    #    product_trend.pop(0)
    return product_loc

#------------------------------------------------
def show_video(stop_event, img_queue, terminal_queue):
    global thread_flag
    cap=cv2.VideoCapture("./product_sale_video/live_video.mp4")
    frame_counter = 0
    if not cap.isOpened():
        print("...can not open video...")
    while thread_flag:#True
        #print("lopppppp")
        ret, frame_v = cap.read()
        if not ret or not thread_flag:
            print("...can not receive frame...")
            cap.release()
            break
        
        ter_flag = terminal_queue.get()
        if ter_flag=="stop":
            print("video_end")
            break
        
        #print("frame_counter",frame_counter)
        
        frame_counter = frame_counter +1 
  
        if frame_counter ==int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1:
            #print("aasss")		
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        img_queue.put(frame_v)     
    #cap.release()            
def show_video3(stop_event, video_queue_trigger, img_queue2, triggerflag_queue):
    global thread_flag
    global hist_video
    trigger_flag = 0
    hist_flag = 1
    print("thread_flag",thread_flag)
    while thread_flag:
        print("get_video")
        
        video_select = video_queue_trigger.get()
        print("video_select",video_select)
        if 	video_select=="stop":
            break
        #--------------------------------
        if len(hist_video)>=1:
            if video_select == hist_video[-1]:
                #print("hist_flag_hist_flag_hist_flag")
                hist_flag =0             
            else:
                hist_flag =1
        if len(hist_video)==0:
            hist_flag =1
        hist_video.append(video_select)
        if len(hist_video)>=10:
            hist_video.pop(0)           
        print("hist_video",hist_video)
        #--------------------------------
        if hist_flag == 1:
            if video_select!=0 and video_select!=1 and video_select!=2:
                video_select =3            	    
            #print("aaaaaaaaa")
            cap=cv2.VideoCapture("./product_sale_video/"+str(video_select)+".mp4")
            if not cap.isOpened():
                print("...can not open video3...")           
                break
            while thread_flag:
                ret, frame_v = cap.read()
                if not ret:
                    print("...can not receive frame...")
                    trigger_flag = 0
                    triggerflag_queue.put(trigger_flag)
                    #cap.release()
                    break
                #print("sssssssssss")		
                trigger_flag = 1
                img_queue2.put(frame_v)
                triggerflag_queue.put(trigger_flag)
def show_video4(stop_event,img_queue, terminal_queue, video_queue_trigger, img_queue2, triggerflag_queue):
    global thread_flag
    global hist_video
    trigger_flag = 0
    hist_flag = 1
    #time_flag=0
    print("thread_flag",thread_flag)
    #--------------------------show_video_live
    cap_live=cv2.VideoCapture("./product_sale_video/live_video.mp4")
    frame_counter = 0
    if not cap_live.isOpened():
        print("...can not open video...")
    #--------------------------
    while thread_flag:
        print("get_video")
        if video_queue_trigger.empty():
            #--------------------------show_video_live
            ret, frame_v = cap_live.read()
            if not ret:
                print("...can not receive frame...")
                cap_live.release()
                break

            ter_flag = terminal_queue.get()
            if ter_flag=="stop":
                print("video_end")
                break
            print("frame_counter",frame_counter)
            frame_counter = frame_counter +1 
            if frame_counter ==int(cap_live.get(cv2.CAP_PROP_FRAME_COUNT))-1:
                print("aasss")		
                frame_counter = 0
                cap_live.set(cv2.CAP_PROP_POS_FRAMES, 0)
            img_queue.put(frame_v)
            #--------------------------
        else:
                   
            video_select = video_queue_trigger.get()
            print("video_select",video_select)
            if 	video_select=="stop":
                break
        #--------------------------------
            if len(hist_video)>=1:
                if video_select == hist_video[-1]:
                    #print("hist_flag_hist_flag_hist_flag")
                    hist_flag =0             
                else:
                    hist_flag =1
            if len(hist_video)==0:
                hist_flag =1
            hist_video.append(video_select)
            if len(hist_video)>=10:
                hist_video.pop(0)           
            print("hist_video",hist_video)
        #--------------------------------
            if hist_flag == 1:
                if video_select!=0 and video_select!=1 and video_select!=2:
                    video_select =3            	    
                #print("aaaaaaaaa")
                		
                cap=cv2.VideoCapture("./product_sale_video/"+str(video_select)+".mp4")
                if not cap.isOpened():
                    print("...can not open video3...")           
                    break
                
                #frame_v = cv2.imread("./product_sale_video/"+str(video_select)+".jpg")	
                while thread_flag:
                    ret, frame_v = cap.read()
                    #time_flag = time_flag + 1
                    if not ret:
                        print("...can not receive frame...")
                        trigger_flag = 0
                        triggerflag_queue.put(trigger_flag)
                        #time_flag = 0
                        #cap.release()
                        break
                    if ter_flag=="stop":
                        print("video_end")
                        break
                    #print("sssssssssss")		
                    trigger_flag = 1
                    img_queue2.put(frame_v)
                    triggerflag_queue.put(trigger_flag)  
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    FPS=CvFpsCalc(buffer_len=10)
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    frame_count=0

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    try:
        #model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        #stride=int(model.stride.max())  # model stride#-------------------------
    except:
        #load_darknet_weights(model, weights)
        load_darknet_weights(model, weights[0])
    model.to(device).eval()
    if half:
        model.half()  # to FP16


    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    #vid_path, vid_writer = None, None
    #-------------------
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #-------------------
    # Get names and colors
    names = load_classes(names)
    #print("names",names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    video_name='video_10'
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out_video = cv2.VideoWriter(video_name + '.mp4',fourcc, 10.0, (int(640),int(480)))#設定影片格式與檔名
    #out_video2 = cv2.VideoWriter(video_name + '_raw.mp4',fourcc, 10.0, (int(640),int(480)))#設定影片格式與檔名


    #----------------new
    '''
    pro_label=["green","kolanuts","coco","crispy","blacks","colas","shreds","peperos","comes","shrimps", \
'vegetjuices','blackteas','cheetoss','cocosticks','teabag','greenteas','oreos','peanutchips','yellowchips','sodas']
    '''
    pro_label = ["balenciagablacks", "balenciagapinks", "chloes"]
    #exist_aver=[[] for _ in range(2)]#在list中創n個list空間(n=2)

    cls_num=len(pro_label)
    same_num=10#同一類商品個數
    for i in pro_label:
        globals()[i]=product(same_num)#字串轉變數 ex: green=product(), kolanuts=product()
    #green=product()
    x_wei=0
    y_hei=0

    global thread_flag
    thread_flag = True 
    cam=vStream(0,640,480)
    #time.sleep(1)  # wait time
    # Run inference
    t0 = time.time()
    #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    #video_show_flag = 0
    #cap_always = cv2.VideoCapture("video_10_raw3.mp4")
    #-------------------------------------video live


    img_queue = Queue()#maxsize=10
    #cap = cv2.VideoCapture("./video_10_raw2.mp4")	
    terminal_queue = Queue()#maxsize=10
    stop_event = threading.Event()
    global hist_video
    hist_video=[]
    video_count_flag=1
    video_count = 0 
    #thread_video=threading.Thread(target=show_video,args=(stop_event, img_queue, terminal_queue))
    #thread_video.daemon=True
    #thread_video.start()
    video_img = np.ones((480,640,3),np.uint8)

    video_queue_trigger = Queue()#maxsize=5
    img_queue2 = Queue()#maxsize=5
    triggerflag_queue = Queue()#maxsize=5
  
 
    #thread_video2=threading.Thread(target=show_video3,args=(stop_event, video_queue_trigger, img_queue2, triggerflag_queue))
    #thread_video2.daemon=True
    #thread_video2.start()  
    video_img_triggger1 = np.ones((480,640,3),np.uint8)
    trigger_flag = 0
    thread_video3=threading.Thread(target=show_video4,args=(stop_event,img_queue, terminal_queue, video_queue_trigger, img_queue2, triggerflag_queue))
    thread_video3.daemon=True
    thread_video3.start() 
    '''
    pro_label_flag=["greenflag","kolanutsflag","cocoflag","crispyflag","blacksflag","colasflag","shredsflag","peperosflag","comesflag","shrimpsflag", \
    'vegetjuicesflag','blackteasflag','cheetossflag','cocosticksflag','teabagflag','greenteasflag','oreosflag','peanutchipsflag','yellowchipsflag','sodasflag']
    '''    
    pro_label_flag=["balenciagablacksflag", "balenciagapinksflag", "chloesflag"]
    for i in pro_label_flag:
        globals()[i]=0
    #-------------------------------------  
    for  frame,frame1 in  cam:                              
    #for path, img, im0s, vid_cap in dataset:
        #frame=cam.getFrame()
        #print(frame)
        #out_video2.write(frame)
        img_concat = np.hstack((frame, frame1))        	
        img_frame=img_concat.copy()
        '''
        if 	frame_count == 20:
            cam.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)#maunal exposure    
            exposure = cam.capture.get(cv2.CAP_PROP_EXPOSURE)
            cam.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        '''
        #print("exposure3",cam.capture.get(cv2.CAP_PROP_EXPOSURE))
        for i in range(cls_num):
            pro_name2= globals()[ pro_label[i] ]
            pro_name2.cls_flag=0

        display_fps=FPS.get()
        
        #----------------

        img_frame= letterbox(img_frame, 640)[0]#stride=32
        
        #----------------
        # Convert
        img_frame = img_frame[:, :, ::-1].transpose(2, 0, 1)
        img_frame= np.ascontiguousarray(img_frame)
        
        img_frame = torch.from_numpy(img_frame).to(device)
        img_frame = img_frame.half() if half else img_frame.float()  # uint8 to fp16/32
        img_frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img_frame.ndimension() == 3:
            img_frame = img_frame.unsqueeze(0)
        #----------------
            
        # Inference
        #t1 = time_synchronized()#pytorch為非同步 (unsynchronized) ，不等待程式執行，就執行下一行，加這行可正確估計time.time()
        pred = model(img_frame, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img_frame, im0s)
        
        #print("pred",pred)

            
        # Process detections
        for i, det in enumerate(pred):  # detections per image


            im0=  img_concat.copy()                                 
            #--------------------------------------
            if det is not None and len(det):
           
                #print("aaaaaaas:",torch.where(det==0,True,False))#torch.where(condition, x, y) ==>condition是條件，成立返回x​，不成立返回y
                #print("aaaaaaaa:",torch.any(torch.where(det==0,True,False)))#若tensor中存在元素为True，返回True; 若所有元素都是False，返回False

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_frame.shape[2:], det[:, :4], im0.shape).round()
                '''
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                '''
                # Write resultsf
                for *xyxy, conf, cls in det:

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                                             
                        #--------------------------------------------product history
                        x_wei=abs(int(xyxy[2])-int(xyxy[0]))
                        y_hei=abs(int(xyxy[1])-int(xyxy[3]))
                        #print("area",x_wei*y_hei,'compare_area:',1/2*(640*480))
                        #if x_wei*y_hei<=1/3*(640*480):
                        if x_wei*y_hei<=(640*480):
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            x_val=int( (int(xyxy[0])+int(xyxy[2]))/2 )
                            y_val=int( (int(xyxy[1])+int(xyxy[3]))/2 )
                            if True:#greengood
                                #print("clssss",cls)
                                pro_name= globals()[ pro_label[int(cls)] ] # pro_name= 變數(ex: pro_label[ int(cls) ]--> green , cls是第?類 )
                                print("pro_label[int(cls)]",pro_label[int(cls)])
                                pro_name.cls_flag = pro_name.cls_flag+1# number of greengoood & kolanut (ex: 0 , 1st , 2 nd)
                                
                                #cls_flag=cls_flag+1
                                flag_k=pro_name.cls_flag

                                pro_name.history[ flag_k -1 ].append([x_val,y_val])#center of product greengood
                                if len(pro_name.history[flag_k-1 ])>=2:
                                    #判斷商品移動狀態,pro_name.trend[flag_k-1] --> 1, -1, 0(result of pro_name.history[ flag_k-1 ])	
                                    pro_name.loc_val[ flag_k-1 ]=product_aver_direction(pro_name.trend[flag_k-1],pro_name.aver_trend[ flag_k-1 ],pro_name.history[ flag_k-1 ])
                                if len(pro_name.history[ flag_k-1 ])>=20:
                                    pro_name.history[ flag_k-1 ].pop(0)
                                                                           
            #--------------------------------------------new
            #判斷商品移動狀態，loc_val --> no:消失，goin:往貨架，back:往消費者
            for i in range(cls_num):
                pro_name2= globals()[ pro_label[i] ]
                if len(det)==0:
                    pro_name2.loc_val[0]="no"
                    pro_name2.loc_val[1]="no"
                else :
                    if pro_name2.cls_flag==0:
                        pro_name2.loc_val[0]="no"
                        pro_name2.loc_val[1]="no"
                    elif pro_name2.cls_flag==1:
                        pro_name2.loc_val[1]="no"

            #----------------
            #判斷商品是否存在
            for i in range(cls_num):
                pro_name2= globals()[ pro_label[i] ]
                if pro_name2.cls_flag==1:
                    pro_name2.exist_val[0]=1
                    pro_name2.exist_val[1]=0
                elif pro_name2.cls_flag==2:
                    pro_name2.exist_val[0]=1
                    pro_name2.exist_val[1]=1
                elif pro_name2.cls_flag==0:
                    pro_name2.exist_val[0]=0
                    pro_name2.exist_val[1]=0
            
            #--------------------------------------------
          
            #---------------------------new
            #print("ill",ill)
            for i in range(cls_num):#每一類別商品
                pro_name2= globals()[ pro_label[i] ]                
                #------------------------
                #商品消失後才計數(+1 or -1)
                pro_name2.exist_aver[0]=moving_aver_exist(pro_name2.exist[0],pro_name2.exist_val[0],15)
                pro_name2.exist_aver[1]=moving_aver_exist(pro_name2.exist[1],pro_name2.exist_val[1],15)
                #------------------------
                #商品取放動作判斷一次，直到exist_aver清空
                #根據商品歷史軌跡判斷移動方向，往消費者計數+1，往貨架計數-1
                if pro_name2.flag_product[0] ==1:
                    product_count_g1=moving_aver_trend_analysis2(pro_name2.trend_analysis[0],pro_name2.trend[0])
                    pro_name2.count=pro_name2.count+product_count_g1#商品計數
                    #print("product_count_g1",product_count_g1)
                    if product_count_g1==1:#判斷商品是否+1
                        #print("kkkkkkkkkkkkkkkkkkkkk")
                        globals()[ pro_label_flag[i]] =1#商品flag=1
                    if abs(product_count_g1) == 1:#判斷商品數量是否變化
                        pro_name2.flag_product[0] =0
                '''		
                if pro_name2.flag_product[1] ==1:                
                    product_count_g2=moving_aver_trend_analysis2(pro_name2.trend_analysis[1],pro_name2.trend[1])
                    pro_name2.count=pro_name2.count+product_count_g2
                    if product_count_g2==1:
                        globals()[ pro_label_flag[i]] =1
                    if abs(product_count_g2) == 1:
                        pro_name2.flag_product[1] =0 
                '''
                if pro_name2.exist_aver[0]==0:#相同類別內的商品1              
                    pro_name2.trend[0]=[]#商品歷史軌跡清空
                    pro_name2.trend_analysis[0]=[]#商品軌跡的moving_average清空
                    pro_name2.flag_product[0] =1
                '''
                if pro_name2.exist_aver[1]==0:#相同類別內的商品2
                    pro_name2.trend[1]=[]
                    pro_name2.trend_analysis[1]=[]
                    pro_name2.flag_product[1] =1
                '''          
            #---------------------------
            for i in range(len(pro_label_flag)):
                #print("io",i) 
                if  globals()[ pro_label_flag[i]]==1:
                    print("video trigger ",i)
                    video_queue_trigger.put(i)#觸發商品對應影片
                    #print("video_queue_trigger",list(video_queue_trigger.queue))		                     
                    globals()[ pro_label_flag[i]]=0              
           	

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))


        if True:
            frame_count=frame_count+1

            #cv2.putText(im0, "gesture_g1:" + str(green.loc_val[0]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#當下時間
            #cv2.putText(im0, "gesture_g2:" + str(green.loc_val[1]), (10, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#當下時間s



            #cv2.putText(im0, "gesture_k1:" + str(kolanuts.loc_val[0]), (10, 180),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#當下時間
            #cv2.putText(im0, "gesture_k2:" + str(kolanuts.loc_val[1]), (10, 210),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#當下時間

            #cv2.putText(im0, "gesture_c1:" + str(coco.loc_val[0]), (10, 240),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#當下時間

            #cv2.putText(im0, "gesture_cr1:" + str(crispy.loc_val[0]), (10, 300),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#當下時間

            cv2.putText(im0, "FPS:" + str(display_fps), (10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)# 90
            cv2.putText(im0, "balenciagablack:" + str(balenciagablacks.count), (10, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#120
            cv2.putText(im0, "balenciagapink:" + str(balenciagapinks.count), (10, 90),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#150
            cv2.putText(im0, "chloe:" + str(chloes.count), (10, 120),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#270            
            '''
            cv2.putText(im0, "greengood:" + str(green.count), (10, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#120
            cv2.putText(im0, "kolanut:" + str(kolanuts.count), (10, 90),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#150
            cv2.putText(im0, "coco:" + str(coco.count), (10, 120),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#270
            cv2.putText(im0, "crispy:" + str(crispy.count), (10, 150),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "blackchip:" + str(blacks.count), (10, 180),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "cola:" + str(colas.count), (10, 210),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "shredsquid:" + str(shreds.count), (10, 240),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "pepero:" + str(peperos.count), (10, 270),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "comenoodles:" + str(comes.count), (10, 300),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "shrimpchip:" + str(shrimps.count), (10, 330),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330

            cv2.putText(im0, "vegetjuice:" + str(vegetjuices.count), (450, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "blacktea:" + str(blackteas.count), (450, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "cheetos:" + str(cheetoss.count), (450, 90),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "cocostick:" + str(cocosticks.count), (450, 120),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "greenteabag:" + str(teabag.count), (450, 150),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "greentea:" + str(greenteas.count), (450, 180),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "oreo:" + str(oreos.count), (450, 210),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "peanutchip:" + str(peanutchips.count), (450, 240),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "yellowchip:" + str(yellowchips.count), (450, 270),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            cv2.putText(im0, "soda:" + str(sodas.count), (450, 300),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, cv2.LINE_AA)#330
            '''            
            cv2.namedWindow('k',cv2.WINDOW_NORMAL)
            cv2.imshow('k', im0)
            #print("im0",im0.shape)
            #out_video.write(im0)

            #cv2.imwrite('./test/test_result/'+str(count)+'.jpg',im0)
            #---------------------demo_video_live
            if img_queue.empty():
                pass
            else:
                video_img = img_queue.get()
            #---------------------video trigger
            if img_queue2.empty():
                pass
            else:
                video_img_triggger = img_queue2.get()
            if triggerflag_queue.empty():
                pass
            else:
                trigger_flag = triggerflag_queue.get()
            #---------------------
            print("trigger_flag:", trigger_flag)
            if trigger_flag ==0:
                cv2.namedWindow("product_introduce",cv2.WINDOW_NORMAL)#cv2.WINDOW_NORMAL--->0 : same
                cv2.setWindowProperty("product_introduce", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)	
                cv2.imshow("product_introduce", video_img)
                video_count = video_count+1
                if video_count>=5 and video_count_flag ==1:
                    video_count_flag =0
                    hist_video*=0
            else:
                cv2.namedWindow("product_introduce",cv2.WINDOW_NORMAL)#cv2.WINDOW_NORMAL--->0 : same
                cv2.setWindowProperty("product_introduce", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)	
                cv2.imshow("product_introduce", video_img_triggger)
                video_count =0
                video_count_flag=1
            #print("video_count_flag",video_count_flag)
            #print("video_count",video_count)
            print("hist_video",hist_video)

            key=cv2.waitKey(1)
            terminal_queue.put("run")
            print('display_fps:',display_fps)
            if key==27:
                #out_video.release()
                #out_video2.release()
                break
	
    cv2.destroyAllWindows()
 
    #print("endddd")
    terminal_queue.put("stop")
    video_queue_trigger.put("stop")  
    cam.stop()
    thread_flag = False
    #stop_event.set()
    #thread_video.join()
    #thread_video2.join()
    
    #thread_video.join()
    #exit(1)

    #cap.release()
    #print("stop_event.set()222222",stop_event.set())
    print("state:",threading.current_thread(),threading.active_count())
    print("ok")
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/yolov4-20230703-class3-7500/weights/best.pt', help='model.pt path(s)')#yolov4.weights './runs/train/yolov4-0107-class20-20000/weights/best.pt'#yolov4-0411-class20-20000-2
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam ./video/110202.mp4
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',default=False, action='store_true', help='class-agnostic NMS')#True:不同類別實現NMS , False:僅在同一類別實現NMS
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_mod_1013.cfg', help='*.cfg path')#default='models/yolov4.cfg
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
