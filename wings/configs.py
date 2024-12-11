CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
OTHER_DEFAULT_IMAGE_TOKEN = [f'<image {i}>' for i in range(10)]

# Log & Print
BEGIN_LINE = '========================************========================'
END_LINE = '------------------------------------------------------------'

DATASET_NAME2PATH = {
    'demo_pretrain': 'data/demo_pretrain.json',
    'demo_finetune': 'data/demo_finetune.json'
}
