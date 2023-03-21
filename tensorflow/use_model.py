import tensorflow as tf

saved_model_path = f"/Users/chris/machine_learning/model/saved_flowers_model_efficientnetv2-xl-21k"
imported = tf.saved_model.load(saved_model_path)
