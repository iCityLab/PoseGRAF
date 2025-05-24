


import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os
import io
from PIL import Image
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from model.PoseGRAF import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(192, 192, 192), radius=2)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(192, 192, 192), radius=2)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    # ax.set_aspect('equal') # works fine in matplotlib==2.2.2
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)



def get_pose2D(path, output_dir, type):
    print('\nGenerating 2D pose...')
    path = 'E:\ESSAY\Pose_Estimate_3D\Essay_referenceAndcode\PoseGRAF\demo\lib\checkpoint\pose_hrnet_w48_384x288.pth'
    keypoints, scores = hrnet_pose(path, det_dim=416, num_peroson=1, gen_output=True, type=type)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successful!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, filename, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # + 5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + filename + '.mp4', fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def img2gif(video_path, name, output_dir, duration=0.25):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # + 5

    # png_files = os.listdir(output_dir + 'pose/')

    image_list = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))

    # png_files.sort()

    import imageio
    # image_list = [os.path.join(output_dir + 'pose/', f) for f in png_files]
    for image_name in image_list:
        # read png files
        frames.append(imageio.imread(image_name))
    # save gif
    print(output_dir + name + '.gif')
    imageio.mimsave(output_dir + name + '.gif', frames, 'GIF', duration=1 / fps)

def crop_image(input_path,output_path,crop_size):
    """
    裁剪图像并保存。
    :param input_path: 输入图像的路径
    :param output_path: 裁剪后图像的保存路径
    :param crop_size: 裁剪区域的大小（正方形边长）
    """
    # 加载图像
    image = Image.open(input_path)

    # 获取图像尺寸
    width, height = image.size

    # 计算裁剪区域的中心点
    center_x = width // 2
    center_y = height // 2

    # 计算裁剪区域的边界
    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    right = left + crop_size
    bottom = top + crop_size

    # 确保裁剪区域不会超出图像边界
    left = max(0, left+30)
    top = max(0, top)
    right = min(width, right+20)
    bottom = min(height, bottom)

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))

    # 保存裁剪后的图像
    cropped_image.save(output_path)


def get_3D_pose_from_image(args, keypoints, i, img, model):
    img_size = img.shape
    ## input frames
    if args.type == 'image':
        input_2D_no = keypoints[i]  # [1, 17, 2]
    else:
        input_2D_no = keypoints[0][i]
        input_2D_no = np.expand_dims(input_2D_no, axis=0)

    # for multi-frame models
    # start = max(0, i - args.pad)
    # end =  min(i + args.pad, len(keypoints[0])-1)
    # input_2D_no = keypoints[0][start:end+1]
    # left_pad, right_pad = 0, 0
    # if input_2D_no.shape[0] != args.frames:
    #     if i < args.pad:
    #         left_pad = args.pad - i
    #     if i > len(keypoints[0]) - args.pad - 1:
    #         right_pad = i + args.pad - (len(keypoints[0]) - 1)
    #     input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[:, :, 0] *= -1
    input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

    input_2D = input_2D[np.newaxis, :, :, :, :]

    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

    N = input_2D.size(0)
    # print('input_2D',input_2D,'shape',input_2D.shape)
    ## estimation
    output_3D_non_flip = model(input_2D[:, 0])
    # print('output_3D_non_flip',output_3D_non_flip)
    output_3D_flip = model(input_2D[:, 1])

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    result = (output_3D_non_flip + output_3D_flip) / 2

    '''data_translate_from_txt'''

    def read_txt_to_list(file_path):
        """
        读取 .txt 文件中的数据，并将其转换为嵌套列表格式
        :param file_path: 文件路径
        :return: 嵌套列表
        """
        data_list = []
        temp_list = []

        # 打开文件并逐行读取
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行首行尾的空白字符，并分割成单独的数字
                numbers = line.strip().split()
                for number in numbers:
                    # 将字符串转换为浮点数，并添加到临时列表中
                    temp_list.append(float(number))
                    # 每当临时列表中有3个数字时，将其添加到主列表中，并清空临时列表
                    if len(temp_list) == 3:
                        data_list.append(temp_list)
                        temp_list = []

        # 如果临时列表中还有剩余数字（不足3个），可以处理或忽略
        if temp_list:
            print("警告：文件中剩余的数字不足3个，无法组成完整的子列表。")

        return data_list

    # 示例：指定文件路径
    # file_path = 'E:\ESSAY\Pose_Estimate_3D\Essay_referenceAndcode\PoseGRAF\demo\data.txt'  # 替换为您的文件路径

    # 调用函数并打印结果
    # result = read_txt_to_list(file_path)
    # # 按照序号挑选点，并将每个坐标值除以1000
    # result = [[coord/1000 for coord in result[i - 1]] for i in selected_indices]
    # result = normalize_data(result)
    # print(result)

    item_np = result.cpu().detach().numpy()
    item_flat = item_np.reshape(-1)
    # np.savetxt('E:\ESSAY\Pose_Estimate_3D\Essay_referenceAndcode\PoseGRAF\demo\data1.txt',item_flat.squeeze(),fmt='%.9f',delimiter=' ',newline=' ')

    # with open(file_path1,'w') as file:
    #     for item in result:
    #         for point in item.squeeze(0):  # 去掉 batch 维度
    #             file.write(f"{point[0].item():.6f}, {point[1].item():.6f}, {point[2].item():.6f}\n")



    # data = [[result]]
    data = result
    output_3D = torch.tensor(data)

    output_3D = output_3D[0:, args.pad].unsqueeze(1)

    output_3D[:, :, 0, :] = 0

    post_out = output_3D[0, 0].cpu().detach().numpy()

    # motif
    # post_out[:, 0] = -post_out[:, 0]  # Modified: 反转X轴以消除镜像效果


    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)

    post_out[:, 2] -= np.min(post_out[:, 2])

    input_2D_no = input_2D_no[args.pad]
    ## 2D
    image = show2Dpose(input_2D_no, copy.deepcopy(img))
    output_dir_2D = output_dir + 'pose2D/'
    os.makedirs(output_dir_2D, exist_ok=True)
    cv2.imwrite(output_dir_2D + str(('%04d' % i)) + '_2D.png', image)

    # ## 3D
    # fig = plt.figure(figsize=(9.6, 5.4))
    # gs = gridspec.GridSpec(1, 2)
    # gs.update(wspace=-0.00, hspace=0.05)
    # ax1 = plt.subplot(gs[0], projection='3d')
    # show3Dpose(post_out, ax1)
    # ax2 = plt.subplot(gs[1], projection='3d')
    # show3Dpose(post_out1, ax2)
    #
    # output_dir_3D = output_dir + 'pose3D/'
    # os.makedirs(output_dir_3D, exist_ok=True)
    # plt.savefig(output_dir_3D + str(('%04d' % i)) + '_3D.png', dpi=300, format='png', bbox_inches='tight')
    # 绘制第一个 3D 图形并保存为图片

    post_out_experiment = post_out



    output_dir_3D = output_dir + 'experiment/'
    fig1 = plt.figure(figsize=(5.4, 5.4))  # 创建一个独立的图形窗口
    ax1 = fig1.add_subplot(111, projection='3d')  # 添加一个 3D 子图
    show3Dpose(post_out_experiment, ax1)  # 绘制 3D 姿态图
    plt.savefig(output_dir_3D + '0001_3D.png', dpi=310, format='png', bbox_inches='tight',pad_inches = 0)  # 保存为图片
    plt.close(fig1)  # 关闭图形窗口
    filename = 'crop_perfect_img'
    input_file = output_dir_3D + '0001_3D.png'
    final_path = os.path.join(output_dir_3D, f"{filename}.png")
    crop_image(input_file,final_path,crop_size=950)


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []
        self.weights = {}  # 将每个子节点的权重存储为子字典，如 {child: {'weight1': w1, 'weight2': w2}}


def build_tree():
    nodes = [TreeNode(i) for i in range(17)]

    # 定义树的结构，包含两个权重
    edges = [
        (0, 1, 0, -1), (1, 2, 1, 1), (2, 3, 2, -1),
        (0, 4, 6, -1), (4, 5, 7, 1), (5, 6, 8, -1),
        (0, 7, 12, -1), (7, 8, 13, 1), (8, 14, 3, -1),
        (14, 15, 4, 1), (15, 16, 5, -1), (8, 9, 14, -1),
        (9, 10, 15, 1), (8, 11, 9, -1), (11, 12, 10, 1), (12, 13, 11, -1)
    ]

    # 构建树结构
    for parent, child, weight1, weight2 in edges:
        nodes[parent].children.append(nodes[child])
        nodes[child].parent = nodes[parent]
        nodes[parent].weights[child] = {'weight1': weight1, 'weight2': weight2}

    return nodes


def get_pose3D(path, output_dir, type='image'):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 1  # frames = 1
    args.pad = (args.frames - 1) // 2
    args.previous_dir = './pre_trained_model'
    args.n_joints, args.out_joints = 17, 17
    args.type = type

    ## Reload
    nodes = build_tree()
    model = Model(nodes, args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of IGANet in './pre_trained_model/'
    # model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]
    model_path = 'E:\ESSAY\Pose_Estimate_3D\Essay_referenceAndcode\PoseGRAF\pre_trained_model\PoseGRAF_17_4813.pth'
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)
    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    ## 3D
    print('\nGenerating 3D pose...')

    if type == "image":
        i = 0
        img = cv2.imread(path)
        print('brfore',img)
        get_3D_pose_from_image(args, keypoints, i, img, model)
        print('after',img)

    if type == "video":
        cap = cv2.VideoCapture(path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(video_length)):
            ret, img = cap.read()
            get_3D_pose_from_image(args, keypoints, i, img, model)

    output_dir_2D = output_dir + 'pose2D/'
    output_dir_3D = output_dir + 'pose3D/'

    print('Generating 3D pose successful!')

    ## all
    image_dir = 'results/'
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # crop
        if image_2d.shape[1] - image_2d.shape[0] > 0:
            edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
            image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)

        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        ## save
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='image', help='input type, onlys support image or video')
    parser.add_argument('--path', type=str, default='demo/images/dance.png', help='the path of your file')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    path = args.path  # file path
    img_path = r'E:\ESSAY\Pose_Estimate_3D\Essay_referenceAndcode\PoseGRAF\demo\images\dance.png'
    # video_path = '/home/ubuntu/Desktop/Pose_estimate/draw_Picture/PoseGRAF/demo/videos/dance3.mp4'
    filename = path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + filename + '/'

    get_pose2D(path, output_dir, args.type)
    get_pose3D(img_path, output_dir, args.type)

    if args.type == "video":
        img2video(path, filename, output_dir)
        img2gif(path, filename, output_dir)

    print('Generating demo successful!')

