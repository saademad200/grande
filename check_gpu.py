import tensorflow as tf
import os
import sys

print("TensorFlow version:", tf.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("NVIDIA_VISIBLE_DEVICES:", os.environ.get('NVIDIA_VISIBLE_DEVICES'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
print("Python version:", sys.version)

print("\nGPU Detection:")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("XLA devices:", tf.config.list_logical_devices('XLA_GPU'))

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("\nGPU is properly detected!")
    
    # Try simple GPU operation to confirm it works
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("\nMatrix multiplication result on GPU:")
        print(c.numpy())
else:
    print("\nNo GPU detected by TensorFlow! Check your container configuration.")
    print("\nPossible solutions:")
    print("1. Make sure nvidia-container-toolkit is installed")
    print("2. Run 'sudo systemctl restart docker' on host")
    print("3. Check that your NVIDIA drivers match CUDA version in container")
