"""
Goal:Benchmark the different hardware's performance and power consumption in deep learning inference
Model:Senet154
Metric:GOP/S, GOP/J(the batch size is 1, 8, 32)
Framework:Pytorch
Hardware:CPU...
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import Dataset, DataLoader
import time
import pandas
import pretrained_models as pm
from thop import profile

class InferenceDataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.data = torch.randn(num, 3, 224, 224) # support img size is 3*224*224

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return len(self.data)

class DLHBenchmark():
    """can benchmark GOPS and GOPJ on different hardware by running different model"""
    def __init__(self, warm_up, infer_epoch, batch_size_list, model_list, hardware_List):
        self.warm_up = warm_up
        self.infer_epoch = infer_epoch
        self.batch_size_list = batch_size_list
        self.model_list = model_list
        self.hardward_list = hardware_List

        max_batch_size = max(self.batch_size_list)
        data_num = max_batch_size * (self.warm_up + self.infer_epoch)
        self.dataset = InferenceDataset(data_num)

    def inference_cpu(self, model_name, batch_size):
        durations = []
        time_sum = 0
        ops = 0

        model = pm.__dict__[model_name]()
        img_test = torch.rand(1, 3, 224, 224)
        macs, params = profile(model, inputs=(img_test))
        op_num = macs * 2

        img_dataloader = DataLoader(dataset = self.dataset,
                                batch_size = batch_size,
                                num_workers = 4)

        loop_num = self.warm_up + self.infer_epoch
        model.eval()
        
        for step, img in enumerate(img_dataloader):
            if step >= loop_num:
                break
            start_time = time.time()
            model(img)
            end_time = time.time()
            if step >= self.warm_up:
                now_durations = (end_time - start_time) * 1000
                durations.append(now_durations)
                time_sum += now_durations
        
        total_img_num = self.infer_epoch * batch_size
        ops = op_num * total_img_num / time_sum

        return durations, ops

    def inference_gpu(self):
        return []

    def inference_tpu(self):
        return []

    def inference_ascend(self):
        return []

    def make_hardware_func(self):
        func_map = {'CPU':self.inference_cpu, 'GPU':self.inference_gpu, 'TPU':self.inference_tpu,
                    'Ascend':self.inference_ascend}

        return func_map

    def bench_opsj(self):
        func_map = self.make_hardware_func()

        for batch_size in self.batch_size_list:
            benchmark_ops = {}

            for hardware_type in self.hardware_list:
                benchmark_durations = {}
                ops_record = {}

                for model_name in self.model_list:
                    durations, ops= func_map[hardware_type](model_name, batch_size)
                    benchmark_durations[model_name] = durations
                    ops_record[model_name] = ops

                file_name = "results/" + str(batch_size) + "_" + hardware_type
                benchmark_durations_new = pandas.DataFrame(benchmark_durations)
                benchmark_durations_new.to_csv(file_name, index = False)
                benchmark_ops[hardware_type] = ops_record

            file_name = "results/final_ops_" + str(batch_size)
            benchmark_ops_new = pandas.DataFrame(file_name, index = False)

    
