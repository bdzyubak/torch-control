
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

from services.ocr_lightning_wrapper import ocr_print
import matplotlib.pyplot as plt


model_name = 'microsoft/trocr-large-printed'
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to('cuda')


image = Image.open(r"C:\Users\illan\Downloads\X51009453804.jpg").convert('RGB')
# This ocr model only works on one line of text. Must crop the image line by line.
# TODO: Adaptive cropping
text_lines = list()
scores = list()
text_start_line = 18
bar_height = 50
for bar_top in range(text_start_line, image.size[1] - bar_height, bar_height):
    print(f"Cropping image line {bar_top} : {bar_top + bar_height}")
    bar = image.crop((0, bar_top, image.size[0], bar_top + bar_height))
    bar.show()
    text, score = ocr_print(bar, processor, model)
    text_lines.append(text)
    scores.append(score)

image.show()
plt.hist(scores)
plt.show()

# Fitlering by scores does not yield good separation between partial/empty lines with gibberish responses and
# successfully extracted text
filtered_text = [text_line for (text_line, score) in zip(text_lines, scores) if score > -2]
