from pathlib import Path


from utils.LLM_pytorch_lighting_wrapper import model_setup
from utils.torch_utils import get_frozen_layers


# Test importing supported models with and without freezing. Import will fail on bad implementation so skipping asserts
# for now
model_save_dir = Path(r"D:\Models\LLM") / Path(__file__).stem
model_save_dir.mkdir(exist_ok=True, parents=True)
model_distilbert, trainer = model_setup(model_save_dir,
                             num_classes=5,
                             model_name="distilbert-base-uncased",
                             freeze_pretrained_params=False)

model_save_dir = Path(r"D:\Models\LLM") / Path(__file__).stem
model_save_dir.mkdir(exist_ok=True, parents=True)
model_distilbert_frozen, trainer = model_setup(model_save_dir,
                                               num_classes=5,
                                               model_name="distilbert-base-uncased",
                                               freeze_pretrained_params=True)

trainable, frozen = get_frozen_layers(model_distilbert_frozen)
trainable.sort()
assert trainable == ['model.classifier.bias', 'model.classifier.weight', 'model.pre_classifier.bias',
                            'model.pre_classifier.weight'], "Failed to keep classifier layers trainable"

assert len(frozen) == 100, "Failed to freeze the right number of layers"


# assert model_distilbert != model_distilbert_frozen
