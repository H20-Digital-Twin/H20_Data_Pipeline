import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
import matplotlib.pyplot as plt


def convert_quat_wxyz_to_xyzw(quat_wxyz):
    """将 [w, x, y, z] 格式的四元数转换为 SciPy 兼容的 [x, y, z, w] 格式"""
    quat_wxyz = np.array(quat_wxyz)
    return np.roll(quat_wxyz, shift=-1)

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp",
        damping: float=5e-1,
        verbose: bool=False,
    ) -> None:

        # 为了代码能独立运行，这里添加了try-except块

        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
 


        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])

        
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
        print("Scale ratio: ", ratio)
        
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] *= ratio
    
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.max_iter = 10
        self.solver = solver
        self.damping = damping
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self.task_errors1 = {}
        self.task_errors2 = {}
        self.comparison_history = {}

        self.setup_retarget_configuration()
        

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        self.tasks1, self.tasks2 = [], []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight or rot_weight:
                task = mink.FrameTask(frame_name, "body", pos_weight, rot_weight, 1)
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(convert_quat_wxyz_to_xyzw(rot_offset))
                self.tasks1.append(task)
                
                # 初始化历史记录，包含位置和旋转
                if body_name not in self.comparison_history:
                    self.comparison_history[body_name] = {
                        'target_pos': [], 'actual_pos': [],
                        'target_rot': [], 'actual_rot': [],
                    }
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight or rot_weight:
                task = mink.FrameTask(frame_name, "body", pos_weight, rot_weight, 1)
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(convert_quat_wxyz_to_xyzw(rot_offset))
                self.tasks2.append(task)
  
    def update_targets(self, human_data, offset_to_ground=False):
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name, task in self.human_body_to_task1.items():
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        if self.use_ik_match_table2:
            for body_name, task in self.human_body_to_task2.items():
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
    def retarget(self, human_data, offset_to_ground=False):
        self.update_targets(human_data, offset_to_ground)

        # 简单的迭代求解
        if self.use_ik_match_table1:
            for _ in range(self.max_iter):
                vel = mink.solve_ik(self.configuration, self.tasks1, 1e-2, self.solver, self.damping)
                self.configuration.integrate_inplace(vel, 1e-2)
        if self.use_ik_match_table2:
             for _ in range(self.max_iter):
                vel = mink.solve_ik(self.configuration, self.tasks2, 1e-2, self.solver, self.damping)
                self.configuration.integrate_inplace(vel, 1e-2)

        # 关键: 更新模型状态以获取正确的 xpos 和 xquat
        mj.mj_forward(self.configuration.model, self.configuration.data)

        # 记录位置和旋转数据
        for body_name, task in self.human_body_to_task1.items():
            if body_name in self.scaled_human_data:
                target_pos, target_rot_wxyz = self.scaled_human_data[body_name]
                frame_name = task.frame_name
                body_id = self.configuration.model.body(frame_name).id
                
                actual_pos = self.configuration.data.xpos[body_id]
                actual_rot_wxyz = self.configuration.data.xquat[body_id]

                history = self.comparison_history[body_name]
                history['target_pos'].append(target_pos.copy())
                history['actual_pos'].append(actual_pos.copy())
                history['target_rot'].append(target_rot_wxyz.copy())
                history['actual_rot'].append(actual_rot_wxyz.copy())
            
        return self.configuration.data.qpos.copy()

    # ==================== 新的、分离的绘图方法 ====================

    def plot_position_results(self, output_filename: str = "ik_position_comparison.pdf"):
        """
        仅绘制目标位置与实际位置的对比图，并将结果保存为文件。
        """
        body_names = list(self.comparison_history.keys())
        if not body_names or not self.comparison_history[body_names[0]]['target_pos']:
            print("没有足够的数据用于绘制位置图。")
            return

        num_bodies = len(body_names)
        fig, axes = plt.subplots(num_bodies, 3, figsize=(18, 5 * num_bodies), squeeze=False)
        # fig.suptitle('IK Position Tracking: Target vs. Actual', fontsize=16)

        for i, body_name in enumerate(body_names):
            history = self.comparison_history[body_name]
            target_poses = np.array(history['target_pos'])
            actual_poses = np.array(history['actual_pos'])
            timesteps = range(len(target_poses))

            coords = ['X', 'Y', 'Z']
            for j, coord in enumerate(coords):
                if body_name != 'spine3':
                    ax = axes[i, j]
                    ax.plot(timesteps, target_poses[:, j], 'r--', label=f'Target {coord}')
                    ax.plot(timesteps, actual_poses[:, j], 'b-', label=f'Actual {coord}')
                    ax.set_title(f'{body_name} - {coord} Axis')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Position (m)')
                    ax.legend()
                    # if coord == 'Z':
                    #     ax.set_ylim(0, 1)  # 如果坐标是'Z'，则设置Y轴范围为 -5 到 5


                    if coord == 'Z':
                        # 1. 找到Z轴上所有数据的最小值和最大值
                        z_min = min(target_poses[:, j].min(), actual_poses[:, j].min())
                        z_max = max(target_poses[:, j].max(), actual_poses[:, j].max())

                        # 2. 计算范围和留白
                        z_range = z_max - z_min
                        # 防止数据完全水平时 z_range 为 0
                        if np.isclose(z_range, 0): 
                            z_range = 0.1 # 给一个默认的小范围

                        padding = z_range * 0.9 

                        # 3. 设置动态的Y轴范围
                        ax.set_ylim(z_min - padding, z_max + padding)

                else:
                    ax = axes[i, j]
                    ax.plot(timesteps, target_poses[:, j], 'r--', label=f'Target {coord}')
                    ax.plot(timesteps, actual_poses[:, j], 'b-', label=f'Actual {coord}')
                    ax.set_title(f'{body_name} - {coord} Axis')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Position (m)')
                    ax.legend()
                    # if coord == 'Z':
                    #     ax.set_ylim(0, 1)  # 如果坐标是'Z'，则设置Y轴范围为 -5 到 5
                    
    
                    if coord == 'Z':
                        # 1. 找到Z轴上所有数据的最小值和最大值
                        z_min = min(target_poses[:, j].min(), actual_poses[:, j].min())
                        z_max = max(target_poses[:, j].max(), actual_poses[:, j].max())
    
                        # 2. 计算范围和留白
                        z_range = z_max - z_min
                        # 防止数据完全水平时 z_range 为 0
                        if np.isclose(z_range, 0): 
                            z_range = 0.1 # 给一个默认的小范围
    
                        padding = z_range * 1.6
    
                        # 3. 设置动态的Y轴范围
                        ax.set_ylim(z_min - padding, z_max + padding)
    
    



                ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename)
        print(f"位置对比图已保存至: {output_filename}")
        plt.close(fig)

    # def plot_orientation_results(self, output_filename: str = "ik_orientation_comparison.png"):
    #     """
    #     仅绘制姿态对比图（欧拉角 + 误差角），并将结果保存为文件。
    #     """
    #     body_names = list(self.comparison_history.keys())
    #     if not body_names or not self.comparison_history[body_names[0]]['target_rot']:
    #         print("没有足够的数据用于绘制姿态图。")
    #         return

    #     num_bodies = len(body_names)
    #     fig, axes = plt.subplots(num_bodies, 4, figsize=(24, 5 * num_bodies), squeeze=False)
    #     # fig.suptitle('IK Orientation Tracking: Target vs. Actual', fontsize=16)

    #     for i, body_name in enumerate(body_names):
    #         history = self.comparison_history[body_name]
    #         timesteps = range(len(history['target_rot']))

    #         # --- 欧拉角对比 ---
    #         target_rots_wxyz = history['target_rot']
    #         actual_rots_wxyz = history['actual_rot']
    #         target_eulers = np.array([R.from_quat(convert_quat_wxyz_to_xyzw(q)).as_euler('xyz', degrees=True) for q in target_rots_wxyz])
    #         actual_eulers = np.array([R.from_quat(convert_quat_wxyz_to_xyzw(q)).as_euler('xyz', degrees=True) for q in actual_rots_wxyz])

    #         euler_labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    #         for j, label in enumerate(euler_labels):
    #             ax = axes[i, j]
    #             ax.plot(timesteps, target_eulers[:, j], 'r--', label=f'Target {label}')
    #             ax.plot(timesteps, actual_eulers[:, j], 'b-', label=f'Actual {label}')
    #             ax.set_ylim(-190, 190); ax.set_yticks(np.arange(-180, 181, 60))
    #             ax.set_title(f'{body_name} - {label}'); ax.set_xlabel('Time Step'); ax.set_ylabel('Angle (deg)')
    #             ax.legend(); ax.grid(True)

    #         # --- 姿态误差角 ---
    #         errors = []
    #         for q_t, q_a in zip(target_rots_wxyz, actual_rots_wxyz):
    #             r_t = R.from_quat(convert_quat_wxyz_to_xyzw(q_t))
    #             r_a = R.from_quat(convert_quat_wxyz_to_xyzw(q_a))
    #             angle_rad = (r_t * r_a.inv()).magnitude()
    #             errors.append(np.rad2deg(angle_rad))

    #         ax = axes[i, 3]
    #         ax.plot(timesteps, errors, 'g-')
    #         ax.set_title(f'{body_name} - Overall Angle Error'); ax.set_xlabel('Time Step'); ax.set_ylabel('Error (deg)')
    #         ax.grid(True)

    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    #     plt.savefig(output_filename)
    #     print(f"姿态对比图已保存至: {output_filename}")
    #     plt.close(fig)


    def plot_orientation_results(self, output_filename: str = "ik_orientation_comparison.png"):
        """
        仅绘制姿态对比图（欧拉角），并将结果保存为文件。
        （已移除误差角的可视化）
        """
        body_names = list(self.comparison_history.keys())
        if not body_names or not self.comparison_history[body_names[0]]['target_rot']:
            print("没有足够的数据用于绘制姿态图。")
            return

        num_bodies = len(body_names)
        # 将子图的列数从 4 改为 3，并相应调整 figsize
        fig, axes = plt.subplots(num_bodies, 3, figsize=(18, 5 * num_bodies), squeeze=False)
        # fig.suptitle('IK Orientation Tracking: Target vs. Actual', fontsize=16)

        for i, body_name in enumerate(body_names):
            history = self.comparison_history[body_name]
            timesteps = range(len(history['target_rot']))

            # --- 欧拉角对比 ---
            target_rots_wxyz = history['target_rot']
            actual_rots_wxyz = history['actual_rot']
            
            # 假设 convert_quat_wxyz_to_xyzw 函数存在
            target_eulers = np.array([R.from_quat(convert_quat_wxyz_to_xyzw(q)).as_euler('xyz', degrees=True) for q in target_rots_wxyz])
            actual_eulers = np.array([R.from_quat(convert_quat_wxyz_to_xyzw(q)).as_euler('xyz', degrees=True) for q in actual_rots_wxyz])

            euler_labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
            for j, label in enumerate(euler_labels):
                ax = axes[i, j]
                ax.plot(timesteps, target_eulers[:, j], 'r--', label=f'Target {label}')
                ax.plot(timesteps, actual_eulers[:, j], 'b-', label=f'Actual {label}')
                ax.set_ylim(-190, 190); ax.set_yticks(np.arange(-180, 181, 60))
                ax.set_title(f'{body_name} - {label}'); ax.set_xlabel('Time Step'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True)

            # --- 已移除姿态误差角的可视化代码 ---

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename)
        print(f"姿态对比图已保存至: {output_filename}")
        plt.close(fig)


    def _convert_numpy_to_list(self, data):
        """递归地将字典/列表中的Numpy数组转换为列表，以便JSON序列化。"""
        if isinstance(data, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._convert_numpy_to_list(i) for i in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def save_history_to_json(self, output_filename: str = "retargeting_history.json"):
        """
        将 comparison_history 字典保存为 JSON 文件。
        """
        # 首先，将所有 numpy 数组转换为列表
        history_for_json = self._convert_numpy_to_list(self.comparison_history)

        with open(output_filename, 'w') as f:
            json.dump(history_for_json, f, indent=4)
        print(f"重定向历史数据已保存至: {output_filename}")
    # ==================== 其他辅助方法 ====================

    def error1(self):
        return np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in self.tasks1]))
    
    def error2(self):
        return np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in self.tasks2]))

    def to_numpy(self, human_data):
        return {k: [np.asarray(d) for d in v] for k, v in human_data.items()}

    def scale_human_data(self, human_data, root_name, scale_table):
        root_pos, root_quat = human_data[root_name]
        scaled_root_pos = scale_table[root_name] * root_pos
        
        scaled_data = {root_name: (scaled_root_pos, root_quat)}
        for name, (pos, quat) in human_data.items():
            if name == root_name or name not in scale_table: continue
            local_pos = (pos - root_pos) * scale_table[name]
            scaled_data[name] = (local_pos + scaled_root_pos, quat)
        return scaled_data
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        offset_data = {}
        for name, (pos, quat_wxyz) in human_data.items():
            r_body = R.from_quat(convert_quat_wxyz_to_xyzw(quat_wxyz))
            r_updated = r_body * rot_offsets[name]
            
            global_pos_offset = r_updated.apply(pos_offsets[name])
            
            offset_data[name] = [
                pos + global_pos_offset,
                np.roll(r_updated.as_quat(), 1)
            ]
        return offset_data
            
    def offset_human_data_to_ground(self, human_data):
        z_coords = [pos[2] for name, (pos, _) in human_data.items() if "Foot" in name or "foot" in name]
        if not z_coords: return human_data

        z_offset = -min(z_coords) + 0.1 # 0.1 is ground_offset
        return {name: [pos + [0,0,z_offset], quat] for name, (pos, quat) in human_data.items()}