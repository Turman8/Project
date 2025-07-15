import tensorflow as tf

# 验证版本及GPU支持
print("TensorFlow 版本:", tf.__version__)  # 应输出 2.10.0
print("GPU 可用状态:", tf.config.list_physical_devices('GPU'))  # 若有GPU则显示设备列表