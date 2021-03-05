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
from torch.utils.data import Dataset, DataLoader
import time
import pandas
import pretrained_models as pm
from ptflops import get_model_complexity_info

def transStr2Float(input_str):
    for c in range(len(input_str)):
        if input_str[c] == 'G':
            return float(input_str[:c])

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
    def __init__(self, warm_up, infer_epoch, batch_size_list, model_list, hardware_info):
        self.warm_up = warm_up
        self.infer_epoch = infer_epoch
        self.batch_size_list = batch_size_list
        self.model_list = model_list
        self.hardware_info = hardware_info

        max_batch_size = max(self.batch_size_list)
        data_num = max_batch_size * (self.warm_up + self.infer_epoch)
        self.dataset = InferenceDataset(data_num)

    def inference_cpu(self, model_name, batch_size):
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        macs, params = get_model_complexity_info(model, (3,224,224), as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
        float_macs = transStr2Float(macs)
        op_num = float_macs * pow(10, 9) * 2

        img_dataloader = DataLoader(dataset = self.dataset,
                                batch_size = batch_size,
                                num_workers = 4)
        loop_num = self.warm_up + self.infer_epoch
        time_sum = 0
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
                time_sum += now_durations / 1000
        
        total_img_num = self.infer_epoch * batch_size
        ops = (op_num * total_img_num / time_sum) * pow(10,-9)
        opj = ops / self.hardware_info["CPU"]

        return durations, ops, opj

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
            benchmark_opj = {}

            for hardware_type in self.hardware_info.keys():
                benchmark_durations = {}
                ops_record = {}
                opj_record = {}

                for model_name in self.model_list:
                    durations, ops, opj= func_map[hardware_type](model_name, batch_size)
                    benchmark_durations[model_name] = durations
                    ops_record[model_name] = ops
                    opj_record[model_name] = opj

                file_name = "results/final/" + str(batch_size) + "_" + hardware_type
                benchmark_durations_new = pandas.DataFrame(benchmark_durations)
                benchmark_durations_new.to_csv(file_name, index = False)
                benchmark_ops[hardware_type] = ops_record
                benchmark_opj[hardware_type] = opj_record

            file_name = "results/final/final_ops_" + str(batch_size)
            benchmark_ops_new = pandas.DataFrame(benchmark_ops)
            benchmark_ops_new.to_csv(file_name, index = False)

            file_name = "results/final/final_opj_" + str(batch_size)
            benchmark_opj_new = pandas.DataFrame(benchmark_opj)
            benchmark_opj_new.to_csv(file_name, index = False)

if __name__ == "__main__":
    warm_up = 5
    infer_epoch = 5
    batch_size_list = [1, 2, 4]
    model_list = ["senet154"]
    hardware_info = {"CPU":15}
    dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
                            model_list, hardware_info)
    dlh_bench.bench_opsj()
    
