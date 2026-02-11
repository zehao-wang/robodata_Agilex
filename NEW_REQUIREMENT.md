# 基于当前一些成功，我需要的改进

我要有一个标定模式的script，用于标定世界坐标的transform。
这个标定模式的context和collect_viser.py是一样的，都是在主从臂都连接的情况下。这个标定模式如下。


用户会通过操纵主臂把 end-effector 依次按顺序移动到现实世界平面上的矩形框的四个点，以这四个点定义出新坐标系下的xy单位平面。之后所有碰撞点的坐标都表示在新坐标系下。记录这个新的坐标系和当前robot base坐标系的偏移，存储在data/world_config.json 内。

collect_viser.py 以及 control_arm.py 都需要适应到这个坐标系下。即要读取这个world_conig.json. 对于 collect_viser.py 在写出数据的时候要把坐标系转换到这个world  corrdinate system.对于control_arm.py, 用户输入的坐标也要当成是world coordinate system 下的，转换到robot base坐标再做规划
