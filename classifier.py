import tensorflow as tf

IMAGE_SIZE = (180, 180)
CLASS_NAMES = ["cat", "dog", "other"]


def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, IMAGE_SIZE)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def load_and_preprocess_image(path: str):
    image = tf.keras.preprocessing.image.load_img(path, target_size=IMAGE_SIZE)
    return preprocess_image(image)


def classify_image(model, image_path: str):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    class_probabilities = prediction[0]
    predicted_class_index = class_probabilities.argmax()
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    result_text = f"Я думаю, що це на {100 * class_probabilities[predicted_class_index]:.2f}% {predicted_class_name}!"
    return result_text
