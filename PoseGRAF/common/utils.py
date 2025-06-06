import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)

    return action_error_sum


def p_mpjpe(predicted, target):  # p2, Procrustes analysis MPJPE
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)  # B,1,3
    muY = np.mean(predicted, axis=1, keepdims=True)  # B,1,3

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))  # B,1,1
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])  # B,17,3
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])  # # B,17,3
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def define_actions(action):
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]:
                          {'p1': AccumLoss(), 'p2': AccumLoss()}
                      for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.4f} {2:>10.4f}".format("Average", mean_error_all['p1'].avg, \
                                                    mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, model, model_name):
    # remove the old model
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)
    return previous_name


def back_to_ori_uv(cropped_uv, bb_box):
    """
    for cropped uv, back to origial uv to help do the uvd->xyz operation
    :return:
    """
    N, T, V, _ = cropped_uv.size()
    uv = (cropped_uv + 1) * (bb_box[:, 2:].view(N, 1, 1, 2) / 2.0) + bb_box[:, 0:2].view(N, 1, 1, 2)
    return uv


def get_uvd2xyz(uvd, gt_3D, cam):
    """
    transfer uvd to xyz

    :param uvd: N*T*V*3 (uv and z channel)
    :param gt_3D: N*T*V*3 (NOTE: V=0 is absolute depth value of root joint)

    :return: root-relative xyz results
    """
    N, T, V, _ = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()  # N*T*V*3
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()  # N*T*V*3
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()  # N*T*V*2

    cam_f_all = cam[..., :2].view(-1, 1, 1, 2).repeat(1, T, V, 1)  # N*T*V*2
    cam_c_all = cam[..., 2:4].view(-1, 1, 1, 2).repeat(1, T, V, 1)  # N*T*V*2

    # change to global
    z_global = dec_out_all[:, :, :, 2]  # N*T*V
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]  # N*T*V
    z_global = z_global.unsqueeze(-1)  # N*T*V*1

    uv = enc_in_all - cam_c_all  # N*T*V*2
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  # N*T*V*2
    xyz_global = torch.cat((xy, z_global), -1)  # N*T*V*3
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))  # N*T*V*3

    return xyz_offset


def sym_penalty(dataset, keypoints, pred_out):
    """
    get penalty for the symmetry of human body
    :return:
    """
    loss_sym = 0
    if dataset == 'h36m':
        if keypoints.startswith('sh'):
            left_bone = [(0, 4), (4, 5), (5, 6), (8, 10), (10, 11), (11, 12)]
            right_bone = [(0, 1), (1, 2), (2, 3), (8, 13), (13, 14), (14, 15)]
        else:
            left_bone = [(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)]
            right_bone = [(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)]
        for (i_left, j_left), (i_right, j_right) in zip(left_bone, right_bone):
            left_part = pred_out[:, :, i_left] - pred_out[:, :, j_left]
            right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym += torch.mean(torch.norm(left_part, dim=- 1) - torch.norm(right_part, dim=- 1))
    elif dataset.startswith('STB'):
        loss_sym = 0
    return loss_sym


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3  # B,J,3
    assert len(camera_params.shape) == 2  # camera_params:[B,1,9]
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]  # B,1,3
    p = camera_params[..., 7:]  # B,1,2

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)  # B,J,2
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)  # B, J, 1

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,  # B,J,1
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)  # B,J,1

    XXX = XX * (radial + tan) + p * r2  # B,J,2

    return f * XXX + c


def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip = model(input_2D_flip)

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D


"caculate_Vector_And_Angle"


def get_edge_points_GPU(edge_collection, point_position, edge_point):
    device = point_position.device  # 获取 point_position 的设备（GPU 或 CPU）
    num = len(edge_collection)
    # print('GET_EDGE_POINTS',point_position)

    # 用 torch.zeros 初始化 edge_Matrix，在 GPU 上运行
    edge_Matrix = torch.zeros((num, num), device=device)

    vector_Feature = {}
    temp = []
    temp2 = []

    # 构建 temp 和 temp2 列表
    for i in range(num):
        for j in range(i + 1, num):
            if edge_collection[i] == edge_collection[j]:
                continue
            temp.append(edge_collection[i])
            temp2.append(edge_collection[j])

    temp_num = len(temp)

    # 初始化 edge_collection1 和 edge_collection2
    edge_collection1 = torch.zeros((len(point_position), temp_num, 2), device=device)
    edge_collection2 = torch.zeros((len(point_position), temp_num, 2), device=device)

    # 使用 torch.tile 替代 numpy 的 np.tile
    edge_collection1 = torch.tensor(temp, device=device).unsqueeze(0).repeat(len(point_position), 1, 1)
    edge_collection2 = torch.tensor(temp2, device=device).unsqueeze(0).repeat(len(point_position), 1, 1)

    # 调用 calculate_angle_between_points，确保它返回的是在 GPU 上的张量
    edge_vector, angle_matrix = calculate_angle_between_points_GPU(edge_collection1, edge_collection2, point_position,
                                                                   edge_point)

    # 返回角度矩阵和边向量
    return angle_matrix, edge_vector


def calculate_angle_between_points_GPU(edge_collection1, edge_collection2, point_position, edge_point):
    device = point_position.device  # 获取 point_position 所在的设备（GPU 或 CPU）

    # 使用 torch 的逻辑或操作
    condition1 = torch.logical_or(edge_collection1[:, :, 0] == edge_collection2[:, :, 0],
                                  edge_collection1[:, :, 0] == edge_collection2[:, :, 1])
    # print('condition1',condition1)
    condition2 = torch.logical_or(edge_collection1[:, :, 1] == edge_collection2[:, :, 0],
                                  edge_collection1[:, :, 1] == edge_collection2[:, :, 1])
    # print('condition2',condition2)
    matrix = torch.logical_or(condition1, condition2)
    # print('matrix',matrix)
    matrix_length = matrix.shape[1]
    # print('length',matrix_length)

    # 判断两个边之间有共同点的有哪些
    # print(f"matrix.shape: {matrix.shape}")
    matrix = matrix[0]
    matrix_true = torch.nonzero(matrix).squeeze()

    edge_collection_length = edge_collection1.shape[0]
    position_shape = point_position.shape
    axis1 = position_shape[0]
    axis2 = position_shape[1]
    axis3 = position_shape[2]

    matrix_position = torch.zeros([axis3, matrix_length, axis1], device=device)

    # 上三角矩阵的索引
    row_indices, col_indices = torch.triu_indices(16, 16, offset=1, device=device)

    # 根据 matrix_true 筛选
    up_coords = torch.stack((row_indices, col_indices), dim=-1)
    bottom_coords = up_coords[:, [1, 0]]
    bottom_coords = bottom_coords[matrix]
    up_coords = up_coords[matrix]

    indices = torch.arange(edge_collection1.shape[0], device=device)[:, None, None]
    # print('point_position',point_position)
    # 使用高级索引获取边的位置
    edge_position_collection1 = point_position[indices, edge_collection1]
    # print('edge_position_collection1',edge_position_collection1)
    edge_position_collection2 = point_position[indices, edge_collection2]
    # 根据 matrix 筛选
    edge_position_collection_filter1 = edge_position_collection1[:, matrix, :]
    # print('edge_position_colletion_filter1',edge_position_collection_filter1)
    edge_position_collection_filter2 = edge_position_collection2[:, matrix, :]
    # edge_position_collection_filter2 = edge_position_collection2[matrix, :, :]
    # print('edge_position_collection_filter2',edge_position_collection_filter2)

    # 计算向量
    edge_Vector1 = edge_position_collection_filter1[:, :, 0, :] - edge_position_collection_filter1[:, :, 1, :]
    edge_Vector2 = edge_position_collection_filter2[:, :, 0, :] - edge_position_collection_filter2[:, :, 1, :]
    # print('edge_value1',edge_Vector1)
    # print('EDGE_VALUE2',edge_Vector2)
    # 点积
    edge_dot_product = torch.sum(edge_Vector1 * edge_Vector2, dim=-1)

    # 计算向量的模长
    magnitude_edge1 = torch.norm(edge_Vector1, dim=2)
    magnitude_edge2 = torch.norm(edge_Vector2, dim=2)

    # 处理边长为零的情况
    indice1 = (magnitude_edge1 == 0)
    indice2 = (magnitude_edge2 == 0)
    intersection_and_indice = torch.logical_and(indice1, indice2)
    intersection_or_indice = torch.logical_or(indice1, indice2)

    magnitude_edge1[indice1] = 1
    magnitude_edge2[indice2] = 1
    product = magnitude_edge1 * magnitude_edge2

    edge_dot_product[intersection_or_indice] = 0
    edge_dot_product[intersection_and_indice] = -1
    # print('edge_dot_product',edge_dot_product)
    # print('product',product)

    # 计算 cos 角度
    cos_angle = edge_dot_product / product
    # print('pre_cos_angle', cos_angle)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # print('cos_angle',cos_angle)

    # 计算角度
    angle_rad = torch.acos(cos_angle)
    # print('angle_rad',angle_rad)
    angle_deg = torch.rad2deg(angle_rad)
    # print('angle_rad',angle_deg)
    # 反向角度
    total_deg = torch.cat((angle_deg, angle_deg), dim=-1)
    total_indices = torch.cat((up_coords, bottom_coords), dim=0)

    # 使用高级索引进行赋值
    batch_indices = torch.arange(axis1, device=device)[:, None]

    # 角度矩阵
    angle_matrix = torch.zeros([axis1, 16, 16], device=device)
    angle_matrix[batch_indices, total_indices[:, 0], total_indices[:, 1]] = total_deg

    # 计算每个边的向量
    vector_indices = torch.arange(axis1, device=device)[:, None, None]
    vector_indices = vector_indices.long()
    edge_point = edge_point.long()
    edge_point_position = point_position[vector_indices, edge_point]
    edge_vector = edge_point_position[:, :, 0, :] - edge_point_position[:, :, 1, :]

    # 正则化角度矩阵
    # normal_angle_matrix = (angle_matrix + 180.0) / (180.0 + 180.0) * 2 - 1
    # angle_matrix1 = (angle_matrix != 0).int()
    normal_angle_matrix = normalize_adjacency_matrix(angle_matrix)
    # normal_angle_matrix1 = normalize_adjacency_matrix(angle_matrix1)
    # normal_angle_matrix1 = normalize_undigraph(angle_matrix1)

    del edge_dot_product

    # return edge_vector, normal_angle_matrix, normal_angle_matrix1
    return edge_vector, normal_angle_matrix


def normalize_adjacency_matrix(A_w):
    """
    对批量加权邻接矩阵进行对称归一化，并添加自环。
    Args:
    A_w (torch.Tensor): 原始加权邻接矩阵，形状为 [B, N, N]。
    Returns:
    torch.Tensor: 归一化后的加权邻接矩阵，形状为 [B, N, N]。
    """
    # 获取维度信息
    B, N, _ = A_w.size()  # B 是批量大小, N 是节点数量

    # 添加自环
    I = torch.eye(N, device=A_w.device).unsqueeze(0).expand(B, -1, -1)  # 单位矩阵，形状 [B, N, N]
    A_w_hat = A_w + I  # 添加自环后的邻接矩阵

    # 计算度矩阵
    D_hat = torch.sum(A_w_hat, dim=2)  # 度矩阵对角元素，形状 [B, N]
    D_hat_inv_sqrt = 1.0 / torch.sqrt(D_hat + 1e-8)  # 防止除以 0
    D_hat_inv_sqrt = D_hat_inv_sqrt.unsqueeze(-1)  # 调整形状为 [B, N, 1]

    # 对称归一化
    A_w_norm = D_hat_inv_sqrt * A_w_hat * D_hat_inv_sqrt.transpose(-1, -2)

    return A_w_norm


# import torch


def normalize_undigraph(A):
    # Ensure A is a tensor and move to GPU if available
    A = A.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()  # Ensure A is float

    # Get the batch size and number of nodes (assumed to be square matrices)
    batch_size, num_node, _ = A.shape

    # Compute the degree matrix Dl (sum of each row of A) across the batch
    Dl = A.sum(dim=2)  # Sum across the columns (dim=2)

    # Initialize the diagonal degree matrix Dn
    Dn = torch.zeros((batch_size, num_node, num_node), device=A.device,
                     dtype=torch.float32)  # [batch_size, num_node, num_node]

    # Fill in the diagonal with the inverse square root of the degrees
    for i in range(num_node):
        # We use broadcasting to apply this across the entire batch
        Dn[:, i, i] = torch.pow(Dl[:, i], -0.5)  # Dn[i, i] = Dl[i]**(-0.5)

    # Perform the normalization: Dn * A * Dn for each batch element
    DAD = torch.bmm(torch.bmm(Dn, A), Dn)  # Batch matrix multiplication

    return DAD

# """
#     对批量加权邻接矩阵进行对称归一化，并添加自环。
#     Args:
#     A_w (torch.Tensor): 原始加权邻接矩阵，形状为 [B, N, N]。
#     Returns:
#     torch.Tensor: 归一化后的加权邻接矩阵，形状为 [B, N, N]。
#     """
#     alpha = 0.9
#     self_loop_weight = 0.5
#     eps = 1e-6
#     # 获取维度信息
#     B, N, _ = A_w.size()
#     I = self_loop_weight * torch.eye(N, device=A_w.device).unsqueeze(0).expand(B, -1, -1)
#     A_w_hat = A_w + I  # 添加自环
#     D_hat = torch.sum(A_w_hat, dim=2)
#     D_hat_inv_sqrt = 1.0 / torch.sqrt(D_hat + eps)
#     D_hat_inv_sqrt = D_hat_inv_sqrt.unsqueeze(-1)
#     A_w_norm = D_hat_inv_sqrt * A_w_hat * D_hat_inv_sqrt.transpose(-1, -2)
#
#     return alpha * A_w_hat + (1 - alpha) * A_w_norm  # 保留部分原始矩阵权重