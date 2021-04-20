"""
benchmark_op.py - Benchmark the different hardware's performance and power consumption in deep learning inference
Goal:Benchmark the different hardware's performance and power consumption in deep learning inference
Model:Senet154
Metric:GOP/S, GOP/J(the batch size is 1, 8, 32)
Framework:Pytorch
Hardware:CPU...
"""
import torch
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

        # 获得输入图片的size
        input_res = (3, 224, 224)
        if model_name == "osnet":
            input_res = (3, 256, 128)
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            input_res = (3, 384, 128)
        elif model_name == "alphapose":
            input_res = (3, 256, 192)
        elif model_name == "st_gcn_net":
            input_res = (3, 256, 14)

        # 获得模型的op
        macs, _ = get_model_complexity_info(model, input_res, as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
        float_macs = transStr2Float(macs)
        op_num = float_macs * pow(10, 9) * 2
        # 先使用paper给的值
        if model_name == "efficientnet_b3":
            op_num = 1.8 * pow(10, 9)
        elif model_name == "osnet":
            op_num = 0.98 * pow(10,9)

        # 获得数据集        
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
        opj = ops / self.hardware_info["cpu"]

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

        # 获得输入图片的size
        input_res = (3, 224, 224)
        if model_name == "osnet":
            input_res = (3, 256, 128)
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            input_res = (3, 384, 128)
        elif model_name == "alphapose":
            input_res = (3, 256, 192)
        elif model_name == "st_gcn_net":
            input_res = (3, 256, 14)

        macs, _ = get_model_complexity_info(model, input_res, as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
        float_macs = transStr2Float(macs)
        op_num = float_macs * pow(10, 9) * 2
        # 先使用paper给的值
        if model_name == "efficientnet_b3":
            op_num = 1.8 * pow(10, 9)
        elif model_name == "osnet":
            op_num = 0.98 * pow(10,9)

        # 获得dataset
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
        opj = ops / self.hardware_info["gpu"]
        
        return durations, ops, opj

    def inference_tpu(self):
        return [], 0, 0

    def inference_npu(self, model_name, batch_size):
        if not torch.npu.is_available():
            print("error!!! you don't have npu")
            return [], 0, 0
            
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        model = model.to("npu:0")

        # 获得输入图片的size
        input_res = (3, 224, 224)
        if model_name == "osnet":
            input_res = (3, 256, 128)
        elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
            input_res = (3, 384, 128)
        elif model_name == "alphapose":
            input_res = (3, 256, 192)
        elif model_name == "st_gcn_net":
            input_res = (3, 256, 14)

        # 获得模型的op数据
        macs, _ = get_model_complexity_info(model, input_res, as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
        float_macs = transStr2Float(macs)
        op_num = float_macs * pow(10, 9) * 2
        # 先使用paper给的值
        if model_name == "efficientnet_b3":
            op_num = 1.8 * pow(10, 9)
        elif model_name == "osnet":
            op_num = 0.98 * pow(10,9)

        # 获得dataset
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
            img = img.to("npu:0")
            if step >= loop_num:
                break
            starter, ender = torch.npu.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            starter.record()
            model(img)
            ender.record()
            torch.npu.synchronize()
            if step >= self.warm_up:
                now_durations = starter.elapsed_time(ender)
                durations.append(now_durations)
                time_sum += now_durations / 1000
        
        total_img_num = self.infer_epoch * batch_size
        ops = (op_num * total_img_num / time_sum) * pow(10,-9)
        opj = ops / self.hardware_info["npu"]
        
        return durations, ops, opj

    def make_hardware_func(self):
        func_map = {'cpu':self.inference_cpu, 'gpu':self.inference_gpu, 'tpu':self.inference_tpu,
                    'npu':self.inference_npu}

        return func_map

    def draw_pic(self, result_list):
        for i in range(len(self.batch_size_list)):
            ops_df_list = []
            opj_df_list = []
            for j in range(len(result_list)):
                file_name_ops = "results/bench/ops_" + str(self.batch_size_list[i]) + "_" + str(result_list[j])
                csvdata = pandas.read_csv(file_name_ops, index_col=0)
                df = pandas.DataFrame(csvdata)
                df_new = df.rename(index = self.model_simple)
                ops_df_list.append(df_new)

                file_name_opj = "results/bench/opj_" + str(self.batch_size_list[i]) + "_" + str(result_list[j])
                csvdata = pandas.read_csv(file_name_opj, index_col=0)
                df = pandas.DataFrame(csvdata)
                df_new = df.rename(index = self.model_simple)
                opj_df_list.append(df_new)

            ops_final_df = pandas.concat(ops_df_list, axis=1)
            ops_final_name = "results/bench/final_ops_" + str(self.batch_size_list[i])
            ops_final_df.to_csv(ops_final_name, index=True)
            fig1_title = "GOP/S compare  " + "batch_size = " + str(self.batch_size_list[i])
            ops_pt = ops_final_df.plot(kind="bar", title=fig1_title, logy=True, rot=0)
            ops_fig = ops_pt.get_figure()
            fig1_name = "results/bench/fig_ops_" + str(self.batch_size_list[i]) + ".jpg"
            ops_fig.savefig(fig1_name)

            opj_final_df = pandas.concat(opj_df_list, axis=1)
            opj_final_name = "results/bench/final_opj_" + str(self.batch_size_list[i])
            opj_final_df.to_csv(opj_final_name, index=True)
            fig2_title = "GOP/J compare  " + "batch_size = " + str(self.batch_size_list[i])
            opj_pt = opj_final_df.plot(kind="bar", title=fig2_title, logy=True, rot=0)
            opj_fig = opj_pt.get_figure()
            fig2_name = "results/bench/fig_opj_" + str(self.batch_size_list[i]) + ".jpg"
            opj_fig.savefig(fig2_name)

        return

    def get_prefix(self):
        hprefix = ""
        for key in self.hardware_info.keys():
            hprefix += key[0]

        return hprefix
        
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

                    file_name = "results/" + str(hardware_type) + "/" + str(batch_size) + "_" + hardware_type
                    benchmark_durations_new = pandas.DataFrame(benchmark_durations)
                    benchmark_durations_new.to_csv(file_name, index = False)
                    benchmark_ops[hardware_type] = ops_record
                    benchmark_opj[hardware_type] = opj_record
                
                hprefix = self.get_prefix()

                file_name_ops = "results/bench/ops_" + str(batch_size) + "_" + hprefix
                benchmark_ops_new = pandas.DataFrame(benchmark_ops)
                benchmark_ops_new.to_csv(file_name_ops)                   

                file_name_opj = "results/bench/opj_" + str(batch_size) + "_" + hprefix
                benchmark_opj_new = pandas.DataFrame(benchmark_opj)
                benchmark_opj_new.to_csv(file_name_opj)
        
        return
                

if __name__ == "__main__":
    # warm_up = 5
    # infer_epoch = 5
    # batch_size_list = [1, 2, 4]

    # model_list = ["senet154", "se_resnext50_32x4d", "efficientnet_b3", "unet", "unetpp",
    #         "mgn", "osnet", "pcb", "baseline", "alphapose", "st_gcn_net", "deeppose"]
    # model_simple = {0:"se154", 1:"se50", 2:"eb3", 3:"unet", 4:"unet++", 
    #         5:"mgn", 6:"osnet", 7:"pcb", 8:"bline", 9:"apose", 10:"stgcn", 11:"dpose"}
    
    # hardware_info = {"cpu":45, "gpu":75}
    
    # dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
    #                         model_list, model_simple, hardware_info)
    # dlh_bench.bench_opsj()

    """
    跑单个硬件
    """
    # warm_up = 2
    # infer_epoch = 2
    # batch_size_list = [1, 2]

    # model_list = ["senet154", "se_resnext50_32x4d"]
    # model_simple = {0:"se154", 1:"se50"}
    
    # hardware_info = {"cpu":15}
    
    # dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
    #                         model_list, model_simple, hardware_info)
    # dlh_bench.bench_opsj() 

    """
    画图功能
    """
    # warm_up = 2
    # infer_epoch = 2
    # batch_size_list = [1, 2]

    # model_list = ["senet154", "se_resnext50_32x4d"]
    # model_simple = {0:"se154", 1:"se50"}
    
    # hardware_info = {"cpu":15, "gpu":20}
    
    # dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
    #                         model_list, model_simple, hardware_info)
    # dlh_bench.draw_pic(["c", "g"])