import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from translate import Translator
translator= Translator(to_lang="zh")

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cpu'  # or cpu
torch.set_default_device(device)

# create model
model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/tzn/Projects/Bunny/checkpoints-phi-3/bunny-phi-3-2mdata',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/tzn/Projects/Bunny/checkpoints-phi-3/bunny-phi-3-2mdata',
    trust_remote_code=True)

# text prompt
prompt = '描述一下这幅图像，并使用中文回答，20字以内'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

import os
imagePath = 'demo/'
imageList = os.listdir(imagePath)
imageList.sort()
for imageName in imageList:
    # image, sample images can be found in images folder
    image = Image.open(imagePath + imageName)
    # print(model.config)
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
    # print(image_tensor.shape)
    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    res = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(res)
    # translation = translator.translate(res)
    # print(translation)