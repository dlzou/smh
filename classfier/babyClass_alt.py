import fastai
from fastai.metrics import accuracy
from fastai.torch_core import *
from fastai_audio import *
from fastai.vision import models, ClassificationInterpretation

DATA = Path('sound-downloader')
NSYNTH_AUDIO = DATA/'nsynth_audio' # contains train and valid folders