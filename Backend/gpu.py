import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # กำหนดการใช้ GPU หนึ่งตัว (คุณสามารถปรับแต่งให้เหมาะสมกับการใช้งานของคุณ)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # จำกัดหน่วยความจำ GPU
    except RuntimeError as e:
        print(e)
