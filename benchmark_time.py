import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import pandas
import pretrained_models as pm

MODEL_LIST = {
    'senet154': pm.__dict__['senet154'](),
}

HARDWARE_LIST = ['CPU']

torch.backends.cudnn.benchmark = True

WARM_UP = 10
NUM_TEST = 20
BATCH_SIZE = 16

"""
this function is used for benchmarking the inference time of model in cpu (or different hardware)
"""
def inference_cpu():
    benchmark = {}
    img = Variable(torch.randn(BATCH_SIZE, 3, 224, 224), volatile = True)
    print("benchmarking model inference time begin...")
    for hardward_type in range(len(HARDWARE_LIST)):
        for model_name, model in MODEL_LIST.items():
            durations = []
            model.eval()
            for step in range(WARM_UP + NUM_TEST):
                if step < WARM_UP:
                    continue
                start_time = time.time()
                model.forward(img)
                end_time = time.time()
                durations.append((start_time - end_time) * 1000)
            benchmark[model_name] = durations
    print("benchmarking end")
    return benchmark

if __name__ == "__main__":
    inference_benchmark = pandas.DataFrame(inference())
    inference_benchmark.to_csv("results/model_inference_benchmark_cpu", index = False)