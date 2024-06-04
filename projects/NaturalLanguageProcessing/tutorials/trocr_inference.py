
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import ViTModel, ViTConfig, TrOCRForCausalLM, TrOCRConfig
from PIL import Image

from ocr_lightning_wrapper import ocr, ocr_print


model_name = 'microsoft/trocr-large-printed'
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to('cuda')


image = Image.open(r"C:\Users\illan\Downloads\X51009453804.jpg").convert('RGB')
# This ocr model only works on one line of text. Must crop the image line by line.
# TODO: Adaptive cropping
text_lines = list()
text_start_line = 18
bar_height = 25
for bar_top in range(text_start_line, image.size[1] - bar_height, bar_height):
    print(f"Cropping image line {bar_top} : {bar_top + bar_height}")
    bar = image.crop((0, bar_top, image.size[0], bar_top + bar_height))
    bar.show()
    text = ocr(bar, processor, model)
    print(text)
    text_lines.append(text)

image.show()
print(text_lines)
