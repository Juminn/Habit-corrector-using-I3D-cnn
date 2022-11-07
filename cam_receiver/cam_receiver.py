import os
import cv2

#from time import time
images_root = ".\\images"


bound = 5
frame_len = 19
def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def save_flows(flows,save_dir,num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    #rescale to 0~255 with the bound setting
    #new_dir="flows"
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    #data_root= "C:\\project\\py-denseflow-master\\n\\zqj\\video_classification\\data\\ucf101"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    #save the flows
    save_x=os.path.join(save_dir, 'x', 'flow_x_{:05d}.jpg'.format(num))
    save_y=os.path.join(save_dir, 'y', 'flow_y_{:05d}.jpg'.format(num))
    #flow_x_img=Image.fromarray(flow_x)
    #flow_y_img=Image.fromarray(flow_y)

    cv2.imwrite(save_x, flow_x)
    cv2.imwrite(save_y, flow_y)
    #imageio.imwrite(save_x,flow_x_img) #내가
    #imageio.imwrite(save_y,flow_y_img) #내가
    return 0

#main stert

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
cap = webcam

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 352)  #기본은 640이엇음
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 288) #기본은 480이엇음


print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame width:', int(cap.get(cv2.CAP_PROP_FPS)))

if not webcam.isOpened():
    print("Could not open webcam")
    exit()


frame_num = 0
dir_num_f = 0
dir_num_n = 0
now_key = -1

#make dir
if not (os.path.isdir(images_root)):
    os.makedirs(images_root)


if not os.path.exists(os.path.join(images_root, "x")):
    os.mkdir(os.path.join(images_root, "x"))
if not os.path.exists(os.path.join(images_root, "y")):
    os.mkdir(os.path.join(images_root, "y"))

# loop through frames
while webcam.isOpened():
   # start_time = time()
    # read frame from webcam
    status, frame = webcam.read()


    #tmp =  cv2.waitKeyEx(1)
    #if tmp != -1:
    #    now_key = tmp

    if not status:
        break

    # display output

    cv2.imshow("captured frames", frame)
    cv2.waitKey(1)

    imagename = os.path.join(images_root, 'img_{:05d}.jpg'.format(frame_num))
    cv2.imwrite( imagename , frame)

    if frame_num == 0:
        prev_image = frame
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)

    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame_0 = prev_gray
    frame_1 = gray

    dtvl1 = cv2.createOptFlow_DualTVL1()
    flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
    save_flows(flowDTVL1, images_root, frame_num, bound)
    prev_gray = gray

    frame_num += 1
    if frame_num >= frame_len:
        del_imagename = os.path.join(images_root, 'img_{:05d}.jpg'.format(frame_num-frame_len))
        del_flowxname = os.path.join(images_root, 'x', 'flow_x_{:05d}.jpg'.format(frame_num-frame_len))
        del_flowyname = os.path.join(images_root, 'y', 'flow_y_{:05d}.jpg'.format(frame_num-frame_len))

        os.remove(del_imagename)
        os.remove(del_flowxname)
        os.remove(del_flowyname)


    # press "Q" to stop
    #if now_key  == 0x720000:
    #    break
  #  end_time = time()
  #  print("Duration: " + str(end_time - start_time ))

# release resources
webcam.release()
cv2.destroyAllWindows()