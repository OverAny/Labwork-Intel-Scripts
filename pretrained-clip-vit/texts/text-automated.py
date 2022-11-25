### https://mmocr.readthedocs.io/en/latest/textdet_models.html ###

from mmocr.utils.ocr import MMOCR
ocr = MMOCR(recog='CRNN', recog_ckpt='crnn_academic-a723a1c5.pth', recog_config='crnn_academic_dataset.py', det='DB_r18', det_ckpt='dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth', det_config='dbnet_r18_fpnc_1200e_icdar2015.py')
# ocr = MMOCR(det='TextSnake', recog=None)
# ocr = MMOCR(det='PS_CTW', recog='SAR', kie='SDMGR')

### DATA ###
filename = 'images/a t-shirt that says “Sorry I’m late. I didn’t want to come.__v4.png'

results = ocr.readtext(filename, output='output/', export='output/')
print(results[0]['text'])

### THINGS TO CONSIDER ###
# does this model work?
# how do you make sure the parts of the text are in order as requested?
# how lenient are you with the accuracy to the text? (might read right as righ even though it is the correct spelling on the image)
