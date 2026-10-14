## 1. gasgap
›             gap_depth=(-0.1, -2.0),
              ground_width_range=(0.5, 0.5), ground_height_max=0.025
          ),
          "hf_steppingstones": HfSteppingStonesTerrainCfg(
              proportion=0.2, stone_height_max=0.05, stone_width_range=(0.25, 0.5), stone_distance_range=(0.1, 0.2), platform_width=2.0,
              holes_depth=(-0.1, -2.0), border_width=0.5
          ), 这里我想实现一个想法,帮我在这两个地形分别中间gap里面（比如gap地形中间的空隙等）,添加一种绿色可穿透的区域（可视化是绿色,实际训练的时候
  不开启就看不见,不影响训练）, 然后这个绿色的区域正好填满这些地形的所有gap. 这个区域的用处我是这么想的,一旦机器人的脚或者身体部位穿透进去,我就给予
  穿透深度和持续时间的惩罚,当然这个是一个奖励函数的写法. 帮我将可穿透区域生层 相关代码写在'/home/ab123456/文档/GO2_unitree_rl_lab/source/
  unitree_rl_lab/unitree_rl_lab/tasks/locomotion/terrains/penalty_gap_gas.py' 这个里面


## 2. 

  › '/home/ab123456/文档/GO2_unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/height_scan/height_scan_grad.py' 在这里帮我实现将高程图
  前后两帧作差,这两个高程图都应该转换到我当前的坐标系中,这主要是为了应对旋转. 详细来说就是上一帧的高程图转换到当前坐标系中,可能是yaw的转换,然后对应scan
  dot的高度进行相减,或者再除以dt表示变化率梯度,来衡量地形变化的向量场. 同时在这里添加'/home/ab123456/文档/GO2_unitree_rl_lab/source/unitree_rl_lab/
  unitree_rl_lab/tasks/locomotion/height_scan/height_scan_visualization.py'这个向量场的可视化,就是在原来的height scan dot的点的基础上,增加颜色的变化,也
  就是从粉色到深红色的变化,颜色越浅,就代表当前的梯度越小,越深代表越大, 可以直接修改原来的height scna 颜色逻辑上面.


## 3. 

 '/home/ab123456/文档/GO2_unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/height_scan/height_scan_feet_waypoint.py' 帮我在这里实现
  一个逻辑,当机器人移动的时候,当机器人heading朝向了目标点,并且自己也有一个这个正确移动方向得线速度时候,认为机器人已经开始尝试正确行走了，这时我扫描前方高
  程图区域大小范围的面积,当扫描到平台时候,靠左边的平台生成红色的目标落足点,靠右边的平台生成蓝色的目标落足点.如果只有一个平台就尽可能交替生成,这个只要方便
  机器人行走即可

## 4.

› '/home/ab123456/文档/GO2_unitree_rl_lab/rsl_rl-3.0.1/rsl_rl/modules/actor_critic_3dcnn_vae.py' 帮我实现一个设计. 在现有基础上,增加一个3D-CNN,输入是过
  去3帧（不一定是连续的3帧,给一个参数可以选择中间隔若干帧选择,从当前帧自动往前推算,比如参数等于2,就是t-2,t-4,t-6等等）以及当前帧的高程图,然后通过3D-CNN
  提取时间特征输出一个latent,然后这个latent就是用来预测未来一帧高程图. 这个预测latent的监督信号,就是rollout下一个状态下,通过原始VAE编码后的高程图,这个
  作为监督信号. 最终将当前高程图VAE输出latent以及3D-CNN根据一段时间高程图进行预测的未来一帧高程图latent 合并到一起,一块儿输入到actor网络中. 其中这个3D-
  CNN网络主要是想要倚靠他学习历史高程图之间变化信息


## 5. 
动态调价DR的网络，自适应，前期不要开的太大，后期根据学习情况渐渐调大