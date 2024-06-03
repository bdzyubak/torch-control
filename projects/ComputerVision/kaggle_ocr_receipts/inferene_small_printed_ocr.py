from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import tqdm

# load image from the IAM database (actually this model is meant to be used on printed text)
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

for model_name in tqdm(['microsoft/trocr-base-stage1', 'microsoft/trocr-large-printed']):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(generated_ids)
    print(generated_text)

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True,  max_new_tokens=10,
                                            min_new_tokens=10)[0]
    print(generated_ids)
    print(generated_text)
