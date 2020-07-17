import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

from image_enhancement.utils.pre_processing_image import decodeImage, decodeImg, preprocess_image
from image_enhancement.utils.painter import Visualizer
from image_enhancement.utils.tools import get_cfg
from image_enhancement import models

# from image_enhancement.models.RRDBNet import RRDBNet
# MODEL_PATH = 'models/GAN/my_final_bike_GAN_model'
# model = RRDBNet(blockNum=10)
# model.load_weights(MODEL_PATH)

models.show_avai_models()
config_path = 'configs/server_api.yaml'
model = models.build_model('RRDBNet', config_path)


def main():
    # DATA_PATH = './Samples'
    # img_path = os.path.join(DATA_PATH, 'LP_15A56744_lowRes.jpg')

    config = get_cfg(config_path)
    img_path = config['input']['image']

    image = decodeImage(img_path)
    img_batch = preprocess_image(image)
    plt.figure()
    plt.imshow(image)
    plt.show()

    painter = Visualizer()
    painter.plot(img_batch, img_batch)

    yPred = model.predict(img_batch)
    painter.plot(img_batch, yPred)

    x = np.squeeze(np.clip(yPred, a_min=0, a_max=1))

    plt.imshow(x)
    plt.show()
    if not os.path.exists("logs/outputs"):
        os.makedirs("logs/outputs")
    print(f"[INFO] Check the result at: logs/outputs")
    plt.imsave("logs/outputs/test.png", x)


if __name__ == '__main__':
    main()