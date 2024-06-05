from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from torch_utils import display_tensor_with_PIL


processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to('cuda')

image_newspaper = Image.open(r"D:\data\tutorials\images\newspaper\image_1.png").convert('RGB')
pixel_values = processor(image_newspaper, return_tensors="pt").pixel_values.to('cuda')
image_newspaper.show()
display_tensor_with_PIL(pixel_values)
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_ids)
print(generated_text)


image1 = Image.open(r"C:\Users\illan\Downloads\X51009453804.jpg").convert('RGB')
pixel_values = processor(image1, return_tensors="pt").pixel_values.to('cuda')
# image1.show()
# display_tensor_with_PIL(pixel_values)
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_ids)
print(generated_text)

# invoice_image1 = image1.crop((0, 200, image1.size[0], 225))
# ocr(invoice_image1, processor, model)
