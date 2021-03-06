"""
benchmark_bs.py - benchmark how the batch size influence the time of inference
"""
from benchmark_time import BATCH_SIZE
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import pandas
import pretrained_models as pm

MODEL = pm.__dict__['senet154']()

WARN_UP = 10
NUM_TEST = 20
MAX_BATCH_SIZE = 128

class InferenceDataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.data = torch.randn(num, 3, 224, 224)

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.num

dataset = InferenceDataset(MAX_BATCH_SIZE * (WARN_UP + NUM_TEST))

def inference_cpu(now_batch_size):
    img_loader = DataLoader(dataset, batch_size=now_batch_size, num_workers=4)
    durations = []

    print("begin!!! now batch_size is ", now_batch_size)
    with torch.no_grad():
        MODEL.eval()
        loop_num = WARN_UP + NUM_TEST
        for step, img in enumerate(img_loader):
            if step >= loop_num:
                break

            try:
                start_time = time.time()
                MODEL(img)
                end_time = time.time()
            except RuntimeError as err:
                print("the batch size is too big !!! ", err)
            except MemoryError as err:
                print("the memory is not enough !!! ", err)

            if step >= WARN_UP:
                durations.append(((end_time - start_time) * 1000))

    return durations

def get_avg_time(inference_durations, batch_size):
    sum = 0
    for i in range(len(inference_durations)):
        sum += inference_durations[i]
    avg_time = sum / (batch_size * (NUM_TEST))
    return avg_time

def test_batch_size_influence(times):
    benchmark = {}
    avg_list = []

    for i in range(times):
        batch_size = 2**i

        inference_durations = inference_cpu(batch_size)
        inference_record = pandas.DataFrame(inference_durations)
        file_name = "results/" + str(batch_size) + "_time"
        inference_record.to_csv(file_name, index = False)
        
        avg_time = get_avg_time(inference_durations, batch_size)
        avg_list.append(avg_time)
        flag = "batch_size_" + str(batch_size)
        benchmark[flag] = avg_list
    
    bs_benchmark = pandas.DataFrame(benchmark)
    bs_benchmark.to_csv("results/bs_benchmark", index = False)

    

if __name__ == "__main__":
    test_batch_size_influence(8)    
    
