"""TODO: Add docstring."""

import os
import time

import numpy as np
import pyarrow as pa
from dora import Node
from piper_sdk import C_PiperInterface


def enable_fun(piper: C_PiperInterface):
    """使能机械臂并检测使能状态,尝试0.05s,如果使能超时则退出程序."""
    enable_flag = False

    timeout = 0.05  # 超时时间（秒）
    interval = 0.01  # 每次轮询间隔（秒）
    
    start_time = time.time()
    while not enable_flag:
        enable_flag = piper.EnablePiper()
        
        print("--------------------")
        print("使能状态:", enable_flag)
        print("--------------------")

        time.sleep(0.01)
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print("Piper机械臂自动使能超时....")
            break


def main():
    """TODO: Add docstring."""
    elapsed_time = time.time()
    can_bus = os.getenv("CAN_BUS", "")
    piper = C_PiperInterface(can_bus)
    piper.ConnectPort()

    factor = 57324.840764  # 1000*180/3.14
    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "action_joint":
                # print(f" get action_joint")
                enable_fun(piper)

                # Do not push to many commands to fast. Limiting it to 30Hz
                if time.time() - elapsed_time > 0.03:
                    elapsed_time = time.time()
                else:
                    continue

                position = event["value"].to_numpy()

                # print(f"action_joint: {position}")
                joint_0 = round(position[0] * factor)
                joint_1 = round(position[1] * factor)
                joint_2 = round(position[2] * factor)
                joint_3 = round(position[3] * factor)
                joint_4 = round(position[4] * factor)
                joint_5 = round(position[5] * factor)

                piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)

            elif event["id"] == "action_endpose":
                enable_fun(piper)

                # Do not push to many commands to fast. Limiting it to 30Hz
                if time.time() - elapsed_time > 0.03:
                    elapsed_time = time.time()
                else:
                    continue

                position = event["value"].to_numpy()
                piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                piper.EndPoseCtrl(
                    position[0] * 1000 * 1000,
                    position[1] * 1000 * 1000,
                    position[2] * 1000 * 1000,
                    position[3] * 1000 / (2 * np.pi) * 360,
                    position[4] * 1000 / (2 * np.pi) * 360,
                    position[5] * 1000 / (2 * np.pi) * 360,
                )
                piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
            
            elif event["id"] == "action_gripper":
                # print(f" get action_gripper")
                enable_fun(piper)

                # Do not push to many commands to fast. Limiting it to 30Hz
                if time.time() - elapsed_time > 0.03:
                    elapsed_time = time.time()
                else:
                    continue

                position = event["value"].to_numpy()
                piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                piper.GripperCtrl(int(abs(position[0] * 1000 * 100)), 1000, 0x01, 0)
                piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)

            else:
                # Slave Arm
                joint = piper.GetArmJointMsgs()

                joint_value = []
                joint_value += [joint.joint_state.joint_1.real / factor]
                joint_value += [joint.joint_state.joint_2.real / factor]
                joint_value += [joint.joint_state.joint_3.real / factor]
                joint_value += [joint.joint_state.joint_4.real / factor]
                joint_value += [joint.joint_state.joint_5.real / factor]
                joint_value += [joint.joint_state.joint_6.real / factor]

                gripper = piper.GetArmGripperMsgs()
                joint_value += [gripper.gripper_state.grippers_angle / 1000 / 100]

                node.send_output("slave_jointstate", pa.array(joint_value, type=pa.float32()))

                position = piper.GetArmEndPoseMsgs()
                position_value = []
                position_value += [position.end_pose.X_axis * 0.001 * 0.001]
                position_value += [position.end_pose.Y_axis * 0.001 * 0.001]
                position_value += [position.end_pose.Z_axis * 0.001 * 0.001]
                position_value += [position.end_pose.RX_axis * 0.001 / 360 * 2 * np.pi]
                position_value += [position.end_pose.RY_axis * 0.001 / 360 * 2 * np.pi]
                position_value += [position.end_pose.RZ_axis * 0.001 / 360 * 2 * np.pi]

                node.send_output("slave_endpose", pa.array(position_value, type=pa.float32()))
                node.send_output(
                    "slave_gripper",
                    pa.array(
                        [gripper.gripper_state.grippers_angle / 1000 / 100],
                        type=pa.float32(),
                    ),
                )

                # Master Arm
                joint = piper.GetArmJointCtrl()

                joint_value = []
                joint_value += [joint.joint_ctrl.joint_1.real / factor]
                joint_value += [joint.joint_ctrl.joint_2.real / factor]
                joint_value += [joint.joint_ctrl.joint_3.real / factor]
                joint_value += [joint.joint_ctrl.joint_4.real / factor]
                joint_value += [joint.joint_ctrl.joint_5.real / factor]
                joint_value += [joint.joint_ctrl.joint_6.real / factor]

                gripper = piper.GetArmGripperCtrl()
                joint_value += [gripper.gripper_ctrl.grippers_angle / 1000 / 100]

                node.send_output("master_jointstate", pa.array(joint_value, type=pa.float32()))

                # position = piper.GetFK(mode="control")
                # position_value = []
                # position_value += [position[5][0] * 0.001]
                # position_value += [position[5][1] * 0.001]
                # position_value += [position[5][2] * 0.001]
                # position_value += [position[5][3] / 360 * 2 * np.pi]
                # position_value += [position[5][4] / 360 * 2 * np.pi]
                # position_value += [position[5][5] / 360 * 2 * np.pi]

                # node.send_output("master_endpose", pa.array(position_value, type=pa.float32()))
                node.send_output(
                    "master_gripper",
                    pa.array(
                        [gripper.gripper_ctrl.grippers_angle / 1000 / 100],
                        type=pa.float32(),
                    ),
                )

        elif event["type"] == "STOP":
            # if not TEACH_MODE:
            #     piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
            #     piper.JointCtrl(0, 0, 0, 0, 0, 0)
            #     piper.GripperCtrl(abs(0), 1000, 0x01, 0)
            #     piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
            # time.sleep(5)
            break


if __name__ == "__main__":
    main()
