from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2
#文本转图片,text to image
pipe = pipeline(task=Tasks.text_to_image_synthesis,
                model='dienstag/vintedois-diffusion-v0-1',
                model_revision='v0.1')

#prompt = 'kneeling cat knight, portrait, finely detailed armor, intricate design, silver, silk, cinematic lighting, 4k'
prompt = 'sun, beach, ocean, sand, sea gull, coconut'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])
