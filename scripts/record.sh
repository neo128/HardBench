read -p "请输入机器人的名称，可输入的有realman、aloha、pika: " robot_type
python ../operating_platform/core/coordinator.py \
  --robot.type=$robot_type