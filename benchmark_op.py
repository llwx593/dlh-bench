"""
benchmark_op.py - Benchmark the different hardware's performance and power consumption in deep learning inference
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

torch.backends.cudnn.benchmark = True

def transStr2Float(input_str):
    for c in range(len(input_str)):
        if input_str[c] == 'G':
            return float(input_str[:c])

class InferenceDataset(Dataset):
    def __init__(self, num, h, w):
        self.num = num
        self.data = torch.randn(num, 3, h, w) # support img size is 3*224*224

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return len(self.data)

class DLHBenchmark():
    """can benchmark GOPS and GOPJ on different hardware by running different model"""
    def __init__(self, warm_up, infer_epoch, batch_size_list, model_list, model_simple, hardware_info):
        self.warm_up = warm_up
        self.infer_epoch = infer_epoch
        self.batch_size_list = batch_size_list
        self.model_list = model_list
        self.model_simple = model_simple
        self.hardware_info = hardware_info

        max_batch_size = max(self.batch_size_list)
        data_num = max_batch_size * (self.warm_up + self.infer_epoch)
        self.dataset = InferenceDataset(data_num, 224, 224)
        self.dataset_reid = InferenceDataset(data_num, 384, 128)
        self.dataset_osnet = InferenceDataset(data_num, 256, 128)
        self.dataset_pose = InferenceDataset(data_num, 256, 192)
        self.dataset_stgcn = InferenceDataset(data_num, 256, 14)

    def inference_cpu(self, model_name, batch_size):
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        input_res = (3, 224, 224)
        if model_name == "osnet":
            input_res = (3, 256, 128)
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            input_res = (3, 384, 128)
        elif model_name == "alphapose":
            input_res = (3, 256, 192)
        elif model_name == "st_gcn_net":
            input_res = (3, 256, 14)
        macs, params = get_model_complexity_info(model, input_res, as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
        float_macs = transStr2Float(macs)
        op_num = float_macs * pow(10, 9) * 2

        # 先使用paper给的值
        if model_name == "efficientnet_b3":
            op_num = 1.8 * pow(10, 9)
        elif model_name == "osnet":
            op_num = 0.98 * pow(10,9)
        
        img_dataset = self.dataset
        if model_name == "osnet":
            img_dataset = self.dataset_osnet
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            img_dataset = self.dataset_reid
        elif model_name == "alphapose":
            img_dataset = self.dataset_pose
        elif model_name == "st_gcn_net":
            img_dataset = self.dataset_stgcn

        img_dataloader = DataLoader(dataset = img_dataset,
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

    def inference_gpu(self, model_name, batch_size):
        if not torch.cuda.is_available():
            print("error!!! you don't have cuda")
            return [], 0, 0
            
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        model = model.to("cuda")
        input_res = (3, 224, 224)
        if model_name == "osnet":
            input_res = (3, 256, 128)
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            input_res = (3, 384, 128)
        elif model_name == "alphapose":
            input_res = (3, 256, 192)
        elif model_name == "st_gcn_net":
            input_res = (3, 256, 14)
        macs, params = get_model_complexity_info(model, input_res, as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
        float_macs = transStr2Float(macs)
        op_num = float_macs * pow(10, 9) * 2

        # 先使用paper给的值
        if model_name == "efficientnet_b3":
            op_num = 1.8 * pow(10, 9)
        elif model_name == "osnet":
            op_num = 0.98 * pow(10,9)

        img_dataset = self.dataset
        if model_name == "osnet":
            img_dataset = self.dataset_osnet
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            img_dataset = self.dataset_reid
        elif model_name == "alphapose":
            img_dataset = self.dataset_pose
        elif model_name == "st_gcn_net":
            img_dataset = self.dataset_stgcn
        img_dataloader = DataLoader(dataset = img_dataset,
                                batch_size = batch_size,
                                num_workers = 4)
        loop_num = self.warm_up + self.infer_epoch
        time_sum = 0
        model.eval()
        
        for step, img in enumerate(img_dataloader):
            img = img.to("cuda")
            if step >= loop_num:
                break
            starter, ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            starter.record()
            model(img)
            ender.record()
            torch.cuda.synchronize()
            if step >= self.warm_up:
                now_durations = starter.elapsed_time(ender)
                durations.append(now_durations)
                time_sum += now_durations / 1000
        
        total_img_num = self.infer_epoch * batch_size
        ops = (op_num * total_img_num / time_sum) * pow(10,-9)
        opj = ops / self.hardware_info["GPU"]
        
        return durations, ops, opj

    def inference_tpu(self):
        return [], 0, 0

    def inference_ascend(self):
        return [], 0, 0

    def make_hardware_func(self):
        func_map = {'CPU':self.inference_cpu, 'GPU':self.inference_gpu, 'TPU':self.inference_tpu,
                    'Ascend':self.inference_ascend}

        return func_map

    def bench_opsj(self):
        func_map = self.make_hardware_func()

        with torch.no_grad():
            for batch_size in self.batch_size_list:
                benchmark_ops = {}
                benchmark_opj = {}

                for hardware_type in self.hardware_info.keys():
                    benchmark_durations = {}
                    ops_record = []
                    opj_record = []

                    for model_name in self.model_list:
                        durations, ops, opj= func_map[hardware_type](model_name, batch_size)
                        benchmark_durations[model_name] = durations
                        ops_record.append(ops)
                        opj_record.append(opj)

                    file_name = "results/final/" + str(batch_size) + "_" + hardware_type
                    benchmark_durations_new = pandas.DataFrame(benchmark_durations)
                    benchmark_durations_new.to_csv(file_name, index = False)
                    benchmark_ops[hardware_type] = ops_record
                    benchmark_opj[hardware_type] = opj_record

                file_name = "results/final/final_ops_" + str(batch_size)
                benchmark_ops_new = pandas.DataFrame(benchmark_ops, index = self.model_simple)
                fig1_title = "GOP/S compare  " + "batch_size = " + str(batch_size)
                ops_pt = benchmark_ops_new.plot(kind="bar", title=fig1_title, logy=True, rot=0)
                ops_fig = ops_pt.get_figure()
                fig1_name = "results/final/fig_ops_" + str(batch_size) + ".jpg"
                ops_fig.savefig(fig1_name)
                benchmark_ops_new.to_csv(file_name, index = True)

                file_name = "results/final/final_opj_" + str(batch_size)
                benchmark_opj_new = pandas.DataFrame(benchmark_opj, index = self.model_simple)
                fig2_title = "GOP/J compare  " + "batch_size = " + str(batch_size)
                opj_pt = benchmark_opj_new.plot(kind="bar", title=fig2_title, logy=True, rot=0)
                opj_fig = opj_pt.get_figure()
                fig2_name = "results/final/fig_opj_" + str(batch_size) + ".jpg"
                opj_fig.savefig(fig2_name)
                benchmark_opj_new.to_csv(file_name, index = True)

if __name__ == "__main__":
    warm_up = 5
    infer_epoch = 5
    batch_size_list = [1, 2, 4]
    # warm_up = 2
    # infer_epoch = 2
    # batch_size_list = [1]
    model_list = ["senet154", "se_resnext50_32x4d", "efficientnet_b3", "unet", "unetpp",
            "mgn", "osnet", "pcb", "baseline", "alphapose", "st_gcn_net", "deeppose"]
    model_simple = ["se154", "se50", "eb3", "unet", "unet++", "mgn", "osnet", "pcb", "bline", "apose", "stgcn", "dpose"]
    hardware_info = {"CPU":45, "GPU":75}
    #hardware_info = {"CPU":15}
    dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
                            model_list, model_simple, hardware_info)
    dlh_bench.bench_opsj()