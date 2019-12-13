# 代码笔记
---

## Bashrc的环境变量设置

```bash
# Virtual Environment Wrapper
alias workoncv-master="source /home/longxiajun/MySoftware/OpenCV/OpenCV-master-py3/bin/activate"
export GIO_EXTRA_MODULES=/usr/lib/x86_64-linux-gnu/gio/modules/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/longxiajun/.mujoco/mujoco150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-418/libGL.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/longxiajun/.mujoco/mujoco200/bin

# RLBench and V-REP
export LD_LIBRARY_PATH=/home/longxiajun/MySoftware/qt/5.12.6/gcc_64/lib:$LD_LIBRARY_PATH
export VREP_ROOT=/home/longxiajun/MySoftware/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VREP_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$VREP_ROOT
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
# export QT_DEBUG_PLUGINS=1  代表是否输出QT的log

# Virtual Environment Wrapper
alias workoncv-master="source /home/longxiajun/MySoftware/OpenCV/OpenCV-master-py3/bin/activate"
export GIO_EXTRA_MODULES=/usr/lib/x86_64-linux-gnu/gio/modules/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/longxiajun/.mujoco/mujoco150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-418/libGL.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/longxiajun/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/longxiajun/.mujoco/mjpro150/bin

# RLBench and V-REP
export VREP_ROOT=/home/longxiajun/MySoftware/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VREP_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$VREP_ROOT
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/longxiajun/MySoftware/qt/5.12.6/gcc_64/lib:$LD_LIBRARY_PATH
# export QT_DEBUG_PLUGINS=1
```

## Pycharm

```bash
PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/home/longxiajun/MySoftware/CoppeliaSim/;VREP_ROOT=/home/longxiajun/MySoftware/CoppeliaSim/
```

