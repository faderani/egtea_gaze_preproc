import glob
import os
import numpy as np
import cv2
import argparse


parser = argparse.ArgumentParser(description='Preprocessing/Visualizing EGTEA Gaze+ gaze annotations')


parser.add_argument('--txtfile', default='./gaze_data', help='path to txt annotations')
parser.add_argument('--datapath', default='dataset', help='path to dataset', required=True)
parser.add_argument('--outputpath', default='output', help='path to output dir', required=True)
parser.add_argument('--p', action='store_false', help='to save data as numpy files')
parser.add_argument('--v', action='store_true', help='to visualize gaze data')




def _str2frame(frame_str, fps=None):
    if fps==None:
        fps = 24

    splited_time = frame_str.split(':')
    assert len(splited_time) == 4

    time_sec = 3600 * int(splited_time[0]) \
               + 60 * int(splited_time[1]) +  int(splited_time[2])

    frame_num = time_sec * fps + int(splited_time[3])

    return frame_num

def parse_gtea_gaze(filename, gaze_resolution=None):
    '''
    Read gaze file in CSV format
    Input:
        name of a gaze csv file
    return
        an array where the each row follows:
        (frame_num): px (0-1), py (0-1), gaze_type
    '''
    if gaze_resolution is None:
        # gaze resolution (default 1280*960)
        gaze_resolution = np.array([960, 1280], dtype=np.float32)

    # load all lines
    lines = [line.rstrip('\n') for line in open(filename)]
    # deal with different version of begaze
    ver = 1
    if '## Number of Samples:' in lines[9]:
        line = lines[9]
        ver = 1
    else:
        line = lines[10]
        ver = 2


    # get the number of samples
    values = line.split()
    num_samples = int(values[4])

    # skip the header
    lines = lines[34:]

    # pre-allocate the array
    # (Note the number of samples in header is not always accurate)
    num_frames = 0
    gaze_data = np.zeros((num_samples*2, 4), dtype=np.float32)

    # parse each line
    for line in lines:
        values = line.split()
        # read gaze_x, gaze_y, gaze_type and frame_number from the file
        if len(values)==7 and ver==1:
            px, py = float(values[3]), float(values[4])
            frame = int(values[5])
            gaze_type = values[6]

        elif len(values)==26 and ver==2:
            px, py = float(values[5]), float(values[6])
            frame = _str2frame(values[-2])
            gaze_type = values[-1]

        else:
            raise ValueError('Format not supported')

        # avg the gaze points if needed
        if gaze_data[frame, 2] > 0:
            gaze_data[frame,0] = (gaze_data[frame,0] + px)/2.0
            gaze_data[frame,1] = (gaze_data[frame,1] + py)/2.0
        else:
            gaze_data[frame,0] = px
            gaze_data[frame,1] = py

        # gaze type
        # 0 untracked (no gaze point available);
        # 1 fixation (pause of gaze);
        # 2 saccade (jump of gaze);
        # 3 unkown (unknown gaze type return by BeGaze);
        # 4 truncated (gaze out of range of the video)
        if gaze_type == 'Fixation':
            gaze_data[frame, 2] = 1
        elif gaze_type == 'Saccade':
            gaze_data[frame, 2] = 2
        else:
            gaze_data[frame, 2] = 3

        num_frames = max(num_frames, frame)

    gaze_data = gaze_data[:num_frames+1, :]

    # post processing:
    # (1) filter out out of bound gaze points
    # (2) normalize gaze into the range of 0-1
    for frame_idx in range(0, num_frames+1):

        px = gaze_data[frame_idx, 0]
        py = gaze_data[frame_idx, 1]
        gaze_type = gaze_data[frame_idx, 2]

        # truncate the gaze points
        if (px < 0 or px > (gaze_resolution[1]-1)) \
           or (py < 0 or py > (gaze_resolution[0]-1)):
            gaze_data[frame_idx, 2] = 4

        px = min(max(0, px), gaze_resolution[1]-1)
        py = min(max(0, py), gaze_resolution[0]-1)

        # normalize the gaze
        gaze_data[frame_idx, 0] = px / gaze_resolution[1]
        gaze_data[frame_idx, 1] = py / gaze_resolution[0]
        gaze_data[frame_idx, 2] = gaze_type

    return gaze_data



# def draw_gaze(gaze_data, dir_path):
#
#
#     all_jpg_files = []
#     for root, dirs, files in os.walk(dir_path):
#             for file in files:
#                 if file.endswith(".jpg"):
#                     all_jpg_files.append(os.path.join(root, file))
#
#
#     for jpg in all_jpg_files:
#         parent_dir = jpg.split("/")[-2]
#         img_idx = int(jpg.split("/")[-1].split(".")[0]) - 1
#         start_frame_num = int(parent_dir.split("-")[-2][1:])
#         #end_frame_num = int(parent_dir.split("-")[-1][1:])
#
#
#
#         img = cv2.imread(jpg)
#
#         h,w,_ = img.shape
#
#
#
#         px = int(gaze_data[start_frame_num + img_idx][0] * img.shape[1])
#         py = int(gaze_data[start_frame_num + img_idx][1] * img.shape[0])
#
#         if int(gaze_data[start_frame_num + img_idx, 2]) == 1:
#             img = cv2.circle(img, (px, py), radius=10, color=(0, 0, 255), thickness=-1)
#
#         dst_path = jpg.replace("./", "./gaze_vis/")
#         dst_path = dst_path.replace(".jpg", ".npy")
#         dst_dir = "/".join(dst_path.split("/")[0:-1])
#         if os.path.exists(dst_dir) == False:
#             os.makedirs(dst_dir)
#         cv2.imwrite(dst_path, img)

def draw_gaze(gaze_data, org_dir_path, dst_dir_path):


    all_jpg_files = []
    for root, dirs, files in os.walk(org_dir_path):
            for file in files:
                if file.endswith(".jpg"):
                    all_jpg_files.append(os.path.join(root, file))


    for jpg in all_jpg_files:
        parent_dir = jpg.split("/")[-2]
        img_idx = int(jpg.split("/")[-1].split(".")[0]) - 1
        start_frame_num = int(parent_dir.split("-")[-2][1:])
        #end_frame_num = int(parent_dir.split("-")[-1][1:])



        img = cv2.imread(jpg)

        h,w,_ = img.shape



        px = int(gaze_data[start_frame_num + img_idx][0] * img.shape[1])
        py = int(gaze_data[start_frame_num + img_idx][1] * img.shape[0])

        if int(gaze_data[start_frame_num + img_idx, 2]) == 1:
            img = cv2.circle(img, (px, py), radius=10, color=(0, 0, 255), thickness=-1)

        dst_path = os.path.join(dst_dir_path, *jpg.split("/")[-2:])
        #dst_path = dst_path.replace(".jpg", ".npy")
        #dst_path = jpg.replace("./", "./gaze_vis/")
        #dst_path = dst_path.replace(".jpg", ".npy")
        dst_dir = "/".join(dst_path.split("/")[0:-1])
        if os.path.exists(dst_dir) == False:
            os.makedirs(dst_dir)

        cv2.imwrite(dst_path, img)

def save_gaze(gaze_data, org_dir_path, dst_dir_path):


    all_jpg_files = []
    for root, dirs, files in os.walk(org_dir_path):
            for file in files:
                if file.endswith(".jpg"):
                    all_jpg_files.append(os.path.join(root, file))


    for jpg in all_jpg_files:
        parent_dir = jpg.split("/")[-2]
        img_idx = int(jpg.split("/")[-1].split(".")[0]) - 1
        start_frame_num = int(parent_dir.split("-")[-2][1:])
        #end_frame_num = int(parent_dir.split("-")[-1][1:])



        img = cv2.imread(jpg)

        h,w,_ = img.shape


        pmap = np.zeros((h,w))

        px = int(gaze_data[start_frame_num + img_idx][0] * img.shape[1])
        py = int(gaze_data[start_frame_num + img_idx][1] * img.shape[0])

        if int(gaze_data[start_frame_num + img_idx, 2]) == 1:
            pmap[py][px] = 1

        dst_path = os.path.join(dst_dir_path, *jpg.split("/")[-2:])
        dst_path = dst_path.replace(".jpg", ".npy")
        #dst_path = jpg.replace("./", "./gaze_vis/")
        #dst_path = dst_path.replace(".jpg", ".npy")
        dst_dir = "/".join(dst_path.split("/")[0:-1])
        if os.path.exists(dst_dir) == False:
            os.makedirs(dst_dir)
        np.save(dst_path,pmap)







if __name__== "__main__":
    args = parser.parse_args()

    all_txt_files = []
    for root, dirs, files in os.walk(args.txtfile):
        for file in files:
            if file.endswith(".txt"):
                all_txt_files.append(os.path.join(root, file))


    for txt in all_txt_files:
        print(txt)
        gaze_data = parse_gtea_gaze(txt)
        org_dir_path = txt.split("/")[-1].split(".")[0]
        dst_dir_path = os.path.join(args.outputpath, org_dir_path)
        org_dir_path = os.path.join(args.datapath, org_dir_path)

        if args.v:
            draw_gaze(gaze_data, org_dir_path, dst_dir_path)
        elif args.p:
            save_gaze(gaze_data, org_dir_path, dst_dir_path)







