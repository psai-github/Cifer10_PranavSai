import numpy as np
import matplotlib.pyplot as plt
from streamlit import *
from PIL import Image
import tensorflow as tf
def main():
    title("Animal Classifier - Pranav Sai")
    write('Upload an image!')
    file = file_uploader("Upload Image",type=["jpg","png"])
    # file = camera_input(label="Upload Picture")
    
    if file:
        print("Hi")

        image1 = Image.open(file)
        image(image1,use_column_width=True)
        resized_image = image1.resize((32,32))
        img_array = np.array(resized_image)/255
        img_array = img_array.reshape((1,32,32,3))
        model = tf.keras.models.load_model("model.h5")
        predictions = model.predict(img_array)
        classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        fig,ax= plt.subplots()
        y_pos = np.arange(len(classes))
        ax.barh(y_pos,predictions[0],align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("Predictions")
        pyplot(fig)
    else:
        text('Please upload an Image')


main()
    