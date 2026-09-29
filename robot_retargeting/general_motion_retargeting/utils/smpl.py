import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data

# def load_smplx_file(smplx_file, smplx_body_model_path):
#     smplx_data = np.load(smplx_file, allow_pickle=True)
#     body_model = smplx.create(
#         smplx_body_model_path,
#         "smplx",
#         gender=str(smplx_data["gender"]),
#         use_pca=False,
#         num_expression_coeffs=0
#     )
    
#     num_frames = smplx_data["pose_body"].shape[0]
#     smplx_output = body_model(
#         betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
#         global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
#         body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
#         transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
#         left_hand_pose=torch.zeros(num_frames, 45).float(),
#         right_hand_pose=torch.zeros(num_frames, 45).float(),
#         jaw_pose=torch.zeros(num_frames, 3).float(),
#         leye_pose=torch.zeros(num_frames, 3).float(),
#         reye_pose=torch.zeros(num_frames, 3).float(),
#         # expression=torch.zeros(num_frames, 10).float(),
#         return_full_pose=True,
#     )
    
#     if len(smplx_data["betas"].shape)==1:
#         human_height = 1.66 + 0.1 * smplx_data["betas"][0]
#     else:
#         human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    
#     return smplx_data, body_model, smplx_output, human_height


# import numpy as np
# import torch
# import smplx

# def load_smplx_file(smplx_file, smplx_body_model_path):
#     """
#     加载SMPLX动作文件并创建SMPLX模型
    
#     参数:
#         smplx_file: SMPLX动作文件路径
#         smplx_body_model_path: SMPLX模型文件目录
        
#     返回:
#         smplx_data: 动作数据
#         body_model: SMPLX模型
#         smplx_output: 模型输出
#         human_height: 实际身高
#     """
#     # 加载数据
#     smplx_data = np.load(smplx_file, allow_pickle=True)
    
#     # 创建SMPLX模型
#     body_model = smplx.create(
#         smplx_body_model_path,
#         "smplx",
#         gender=str(smplx_data["gender"]),
#         use_pca=False,
#     )
    
#     # 获取帧数
#     num_frames = smplx_data["pose_body"].shape[0]
    
#     # 处理betas张量 - 确保与帧数匹配
#     betas_tensor = torch.tensor(smplx_data["betas"]).float()
#     if len(betas_tensor.shape) == 1:
#         # 如果betas是向量，扩展为匹配帧数
#         betas_tensor = betas_tensor.repeat(num_frames, 1)
#     else:
#         # 如果已经是矩阵，确保第一维度匹配帧数
#         if betas_tensor.shape[0] != num_frames:
#             betas_tensor = betas_tensor[:num_frames]
    
#     # 创建表达式张量
#     expression_tensor = torch.zeros(num_frames, 10).float()
    
#     # 创建模型输出
#     smplx_output = body_model(
#         betas=betas_tensor,
#         global_orient=torch.tensor(smplx_data["root_orient"]).float(),
#         body_pose=torch.tensor(smplx_data["pose_body"]).float(),
#         transl=torch.tensor(smplx_data["trans"]).float(),
#         left_hand_pose=torch.zeros(num_frames, 45).float(),
#         right_hand_pose=torch.zeros(num_frames, 45).float(),
#         jaw_pose=torch.zeros(num_frames, 3).float(),
#         leye_pose=torch.zeros(num_frames, 3).float(),
#         reye_pose=torch.zeros(num_frames, 3).float(),
#         expression=expression_tensor,  # 添加表达式参数
#         return_full_pose=True,
#     )
    
#     # 计算实际身高
#     if len(smplx_data["betas"].shape) == 1:
#         human_height = 1.66 + 0.1 * smplx_data["betas"][0]
#     else:
#         human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    
#     return smplx_data, body_model, smplx_output, human_height
def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    
    num_frames = smplx_data["pose_body"].shape[0]

    # 创建 SMPL-X 模型，num_expression_coeffs 设置为 10
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
        num_expression_coeffs=10,  # ← 修改
        num_betas=int(smplx_data["num_betas"])
    )

    # betas 和 expression 扩展到每帧
    betas_tensor = torch.tensor(smplx_data["betas"]).float().unsqueeze(0).expand(num_frames, -1)
    expression_tensor = torch.zeros(num_frames, 10).float()  # 10 表情系数，batch 对齐

    #wu change

    # global_orient = torch.tensor(smplx_data["root_orient"]).float()
    
    # rotation_angle = np.pi  # 180度
    # rotation_matrix = torch.tensor([
    #     [1, 0, 0],
    #     [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
    #     [0, np.sin(rotation_angle), np.cos(rotation_angle)]
    # ], dtype=torch.float32)
    
    # # 应用旋转到全局方向
    # rotated_global_orient = torch.einsum('ij,fj->fi', rotation_matrix, global_orient)

    #wu changed

    # 调用模型
    smplx_output = body_model(
        betas=betas_tensor,
        expression=expression_tensor,
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),  #wu changed
        # global_orient=rotated_global_orient,
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        # body_pose=torch.tensor(smplx_data["pose_body"][100]).float().unsqueeze(0).expand(num_frames, -1),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )

    # 简单估算身高
    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        rot = rot.as_quat
        rot = np.roll(rot, 1)
        # result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))
        result[joint_name] = (joints[i], rot)

  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            rot = rot.as_quat()
            rot = np.roll(rot, 1)
            result[joint_name] = (single_joints[i], rot)
            
            


        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps




