import gradio as gr
import gradio.components as gc
from fastai.vision.all import *
import skimage

learn = load_learner("export.pkl")
labels = learn.dls.vocab

def predict(img):
    # img = PILImage.create(str(img))
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Bear Classifier"
description = "A bear classifier which classifies images to Grizzly, Black and Teddy Bears. \n The model is transfer learning based resnet18 trained on random internet images of the bears."
examples = ["BlackBear.jpg"]
interpretation = "default"
enable_queue = True

gr.Interface(
    fn=predict,
    inputs=gc.Image(shape=(512, 512)),
    outputs=gc.Label(num_top_classes=3),
    title=title,
    description=description,
    examples=examples,
    interpretation=interpretation,
).launch(enable_queue=True)