import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import pandas
import pretrained_models as pm

torch.backends.cudnn.benchmark = True

MODEL_LIST = {
    'senet154': pm.__dict__['senet154']()
}

HARDWARE_LIST = ['CPU', 'GPU']

WARM_UP = 10
NUM_TEST = 20
BATCH_SIZE = 16

"""
the dataset is used for inference of model
"""
class InferenceDataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.data = torch.randn(num, 3, 224, 224)
    
    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.num

img_loader = DataLoader(dataset = InferenceDataset(BATCH_SIZE * (WARM_UP + NUM_TEST)), 
                    batch_size = BATCH_SIZE, 
                    num_workers = 8)

"""
this function is used for benchmarking the inference time of model in cpu (or different hardware)
"""
def inference_cpu():
    benchmark = {}
    print("------ CPU BENCH ------")
    print("Begin benchmarking the time of model inference ...")
    with torch.no_grad():
        for model_name, model in MODEL_LIST.items():
            print("now model is:", model_name)
            durations = []
            model.eval()
            for step, img in enumerate(img_loader):
                start_time = time.time()
                model(img)
                end_time = time.time()
                if step >= WARM_UP:
                    durations.append((end_time - start_time) * 1000)
            benchmark[model_name] = durations
    print("------ END ------")
    return benchmark

"""
this function is used for benchmarking the inference time of model in gpu
"""
def inference_gpu():
    benchmark = {}
    print("------ GPU BENCH ------")
    print("Begin benchmarking the time of model inference ...")
    with torch.no_grad():
        for model_name, model in MODEL_LIST.items():
            durations = []
            model = model.to('cuda')
            model.eval()
            for step, img in enumerate(img_loader):
                img = img.to('cuda')
                torch.cuda.synchronize()
                start_time = time.time()
                model(img)
                torch.cuda.synchronize()
                end_time = time.time()
                if step >= WARM_UP:
                    durations.append((end_time - start_time) * 1000)
            benchmark[model_name] = durations
    print("------ END ------")
    return benchmark

if __name__ == "__main__":
    inference_benchmark_cpu = pandas.DataFrame(inference_cpu())
    inference_benchmark_cpu.to_csv("results/model_inference_benchmark_cpu", index = False)

    #inference_benchmark_gpu = pandas.DataFrame(inference_gpu())
    #inference_benchmark_gpu.to_csv("results/model_inference_benchmark_gpu", index = False)

