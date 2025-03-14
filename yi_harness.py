
import os
import argparse
import sys
import torch
from pprint import pprint
from model import load_llama_model
from LMClass import *

# تقليل التحذيرات الخاصة بـ TensorFlow و NVIDIA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# التأكد من توفر CUDA
if not torch.cuda.is_available():
    print("CUDA is needed to run the model.")
    sys.exit(0)

# إعداد المحلل البرمجي للمدخلات
parser = argparse.ArgumentParser("Run harness evaluation with low-bit Yi models.")
parser.add_argument("-s", "--model-size", choices=["6b", "6B", "34b", "34B"], required=False, default="34B", type=str, help="Which model size to use.")
parser.add_argument("-b", "--wbits", choices=[2, 4], required=False, default=2, type=int, help="Which weight bit to evaluate")
parser.add_argument("-g", "--groupsize", choices=[8, 32], required=False, default=8, type=int, help="Specify quantization groups")
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()
args.model_size = args.model_size.lower()

# إعداد مسار النموذج
model_uri = f'GreenBitAI/yi-{args.model_size}-w{args.wbits}a16g{args.groupsize}'

# إعدادات التحميل
asym = False
bits = args.wbits
double_groupsize = 32
kquant = True if bits == 2 else False
v1 = False
return_config = True
cache_dir = './cache'

# تحميل النموذج والمحلل اللغوي
model, tokenizer, config = load_llama_model(
    model_uri,
    cache_dir=cache_dir,
    groupsize=args.groupsize,
    double_groupsize=double_groupsize,
    bits=bits,
    half=True,
    v1=v1,
    asym=asym,
    kquant=kquant,
    return_config=return_config
)
model.eval()

# إعداد فئة LMClass
lm = LMClass(model_uri, batch_size=args.batch_size, cache_dir=cache_dir)
lm.model = model
lm.tokenizer = tokenizer
lm.config = config
lm.reinitial()

print("✅ Model loaded successfully. Running inference...")

# تنفيذ الاستدلال (Inference) بسؤال واحد
prompt = "Who is Napoleon Bonaparte?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    output = model.generate(
        input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9, top_k=50
    )
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n🤖 الإجابة: ", result)

# تحرير الذاكرة بعد التشغيل
import gc
torch.cuda.empty_cache()
gc.collect()
