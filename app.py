import os
import tensorflow as tf
from flask import Flask, request, render_template
from classifier import classify_image
import time
import threading


app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = "static/uploads/"

MAX_FILE_AGE = 5 * 60


def delete_old_files():
    while True:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = time.time() - os.path.getctime(file_path)
                if file_age > MAX_FILE_AGE:
                    os.remove(file_path)
        time.sleep(60)  # Перевіряємо раз в хвилину


# Запустити фоновий потік для видалення старих файлів
delete_thread = threading.Thread(target=delete_old_files)
delete_thread.daemon = True
delete_thread.start()

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "bird_cat_dog_final_save_at_68.h5")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST"])
def upload_file():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)
    result = classify_image(cnn_model, upload_image_path)
    return render_template("classify.html", image_path=upload_image_path, result=result)


@app.route("/classify", methods=["GET"])
def classify_get():
    return render_template("classify.html")


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
