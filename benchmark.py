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

op_dir = {"senet154": 41.64, "se_resnext50_32x4d": 8.56, "efficientnet_b3": 1.8, "unet": 61.42, "unetpp": 53.06, "mgn": 23.88, "osnet": 
0.98, "pcb": 8.08, "baseline": 8.08, "alphapose": 11.82, "st_gcn_net": 7.8, "deeppose": 8.24}

sop_dir = {"matmul256":0, "matmul1024":0, "matmul4096":0, "cnsmall":0, "cnmid":0, "cnbig":0}

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

    def inference_cpu(self, model_name, batch_size, bench_flag="op"):
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        op_num = op_dir[model_name]

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
        ops = (op_num * total_img_num / time_sum)
        opj = ops / self.hardware_info["cpu"]

        sample_s = total_img_num / time_sum
        sample_j = sample_s / self.hardware_info["cpu"]

        if bench_flag == "op":
            return durations, ops, opj
        elif bench_flag == "sample":
            return sample_s, sample_j

    def inference_gpu(self, model_name, batch_size, bench_flag="op"):
        if not torch.cuda.is_available():
            print("error!!! you don't have cuda")
            return [], 0, 0
            
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        model = model.to("cuda:1")
        op_num = op_dir[model_name]

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
            img = img.to("cuda:1")
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
        ops = (op_num * total_img_num / time_sum)
        opj = ops / self.hardware_info["gpu"]
        
        sample_s = total_img_num / time_sum
        sample_j = sample_s / self.hardware_info["gpu"]

        if bench_flag == "op":
            return durations, ops, opj
        elif bench_flag == "sample":
            return sample_s, sample_j

    def inference_tpu(self):
        return [], 0, 0

    def inference_npu(self, model_name, batch_size, bench_flag="op"):
        if not torch.npu.is_available():
            print("error!!! you don't have npu")
            return [], 0, 0
            
        durations = []
        ops = 0
        opj = 0

        model = pm.__dict__[model_name]()
        model = model.to("npu:0")
        op_num = op_dir[model_name]

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
        ops = (op_num * total_img_num / time_sum)
        opj = ops / self.hardware_info["npu"]

        sample_s = total_img_num / time_sum
        sample_j = sample_s / self.hardware_info["npu"]

        if bench_flag == "op":
            return durations, ops, opj
        elif bench_flag == "sample":
            return sample_s, sample_j

    def micro_cpu(self, model, img):
        start_time = time.time()
        outputs = model(img)
        end_time = time.time()
        durations = (end_time - start_time) * 1000
        return durations

    def micro_gpu(self, model, img):
        starter, ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        starter.record()
        model(img)
        ender.record()
        torch.cuda.synchronize()
        durations = starter.elapsed_time(ender)
        return durations

    def micro_npu(self, model, img):
        starter, ender = torch.npu.Event(enable_timing = True), torch.npu.Event(enable_timing = True)
        starter.record()
        model(img)
        ender.record()
        torch.npu.synchronize()
        durations = starter.elapsed_time(ender)
        return durations

    def make_hardware_func(self):
        func_map = {'cpu':self.inference_cpu, 'gpu':self.inference_gpu, 'tpu':self.inference_tpu,
                    'npu':self.inference_npu}

        return func_map

    def draw_pic(self, result_list, draw_flag="op"):
        for i in range(len(self.batch_size_list)):
            if draw_flag == "op":
                ops_df_list = []
                opj_df_list = []
            elif draw_flag == "sample":
                samples_df_list = []
                samplej_df_list = []
            for j in range(len(result_list)):
                if draw_flag == "op":
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
                elif draw_flag == "sample":
                    file_name_samples = "results/bench/samples_" + str(self.batch_size_list[i]) + "_" + str(result_list[j])
                    csvdata = pandas.read_csv(file_name_samples, index_col=0)
                    df = pandas.DataFrame(csvdata)
                    df_new = df.rename(index = self.model_simple)
                    samples_df_list.append(df_new)

                    file_name_samplej = "results/bench/samples_" + str(self.batch_size_list[i]) + "_" + str(result_list[j])
                    csvdata = pandas.read_csv(file_name_samplej, index_col=0)
                    df = pandas.DataFrame(csvdata)
                    df_new = df.rename(index = self.model_simple)
                    samplej_df_list.append(df_new)
            if draw_flag == "op":
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
            elif draw_flag == "sample":
                samples_final_df = pandas.concat(samples_df_list, axis=1)
                samples_final_name = "results/bench/final_samples_" + str(self.batch_size_list[i])
                samples_final_df.to_csv(samples_final_name, index=True)
                fig3_title = "SAMPLE/S compare  " + "batch_size = " + str(self.batch_size_list[i])
                samples_pt = samples_final_df.plot(kind="bar", title=fig3_title, logy=True, rot=0)
                samples_fig = samples_pt.get_figure()
                fig3_name = "results/bench/fig_samples_" + str(self.batch_size_list[i]) + ".jpg"
                samples_fig.savefig(fig3_name)

                samplej_final_df = pandas.concat(samplej_df_list, axis=1)
                samplej_final_name = "results/bench/final_samplej_" + str(self.batch_size_list[i])
                samplej_final_df.to_csv(samplej_final_name, index=True)
                fig4_title = "SAMPLE/J compare  " + "batch_size = " + str(self.batch_size_list[i])
                samplej_pt = samplej_final_df.plot(kind="bar", title=fig4_title, logy=True, rot=0)
                samplej_fig = samplej_pt.get_figure()
                fig4_name = "results/bench/fig_samplej_" + str(self.batch_size_list[i]) + ".jpg"
                samplej_fig.savefig(fig4_name)

        return

    def draw_op_pic(self, result_list, index_list):
        mat_ops_df_list = []
        mat_opj_df_list = []
        conv_ops_df_list = []
        conv_opj_df_list = []
        for i in range(len(result_list)):
            file_mat_ops = "results/bench/mat_ops_" + str(result_list[i])
            csvdata = pandas.read_csv(file_mat_ops, index_col=0)
            df = pandas.DataFrame(csvdata)
            df_new = df.rename(index = index_list)
            mat_ops_df_list.append(df_new)

            file_mat_opj = "results/bench/mat_opj_" + str(result_list[i])
            csvdata = pandas.read_csv(file_mat_opj, index_col=0)
            df = pandas.DataFrame(csvdata)
            df_new = df.rename(index = index_list)
            mat_opj_df_list.append(df_new)

            file_conv_ops = "results/bench/conv_ops_" + str(result_list[i])
            csvdata = pandas.read_csv(file_conv_ops, index_col=0)
            df = pandas.DataFrame(csvdata)
            df_new = df.rename(index = index_list)
            conv_ops_df_list.append(df_new)

            file_conv_opj = "results/bench/mat_opj_" + str(result_list[i])
            csvdata = pandas.read_csv(file_conv_opj, index_col=0)
            df = pandas.DataFrame(csvdata)
            df_new = df.rename(index = index_list)
            conv_opj_df_list.append(df_new)

        mat_ops_final_df = pandas.concat(mat_ops_df_list, axis=1)
        mat_ops_final_name = "results/bench/mat_final_ops"
        mat_ops_final_df.to_csv(mat_ops_final_name, index=True)
        fig1_title = " matmul op - GOP/S compare  "
        mat_ops_pt = mat_ops_final_df.plot(kind="bar", title=fig1_title, logy=True, rot=0)
        mat_ops_fig = mat_ops_pt.get_figure()
        fig1_name = "results/bench/fig_mat_ops" + ".jpg"
        mat_ops_fig.savefig(fig1_name)

        mat_opj_final_df = pandas.concat(mat_opj_df_list, axis=1)
        mat_opj_final_name = "results/bench/mat_final_opj"
        mat_opj_final_df.to_csv(mat_opj_final_name, index=True)
        fig2_title = " matmul op - GOP/J compare  "
        mat_opj_pt = mat_opj_final_df.plot(kind="bar", title=fig2_title, logy=True, rot=0)
        mat_opj_fig = mat_opj_pt.get_figure()
        fig2_name = "results/bench/fig_mat_opj" + ".jpg"
        mat_opj_fig.savefig(fig2_name)

        conv_ops_final_df = pandas.concat(conv_ops_df_list, axis=1)
        conv_ops_final_name = "results/bench/conv_final_ops"
        conv_ops_final_df.to_csv(conv_ops_final_name, index=True)
        fig3_title = " conv op - GOP/S compare  "
        conv_ops_pt = conv_ops_final_df.plot(kind="bar", title=fig3_title, logy=True, rot=0)
        conv_ops_fig = conv_ops_pt.get_figure()
        fig3_name = "results/bench/fig_mat_ops" + ".jpg"
        conv_ops_fig.savefig(fig3_name)

        conv_opj_final_df = pandas.concat(conv_opj_df_list, axis=1)
        conv_opj_final_name = "results/bench/conv_final_opj"
        conv_opj_final_df.to_csv(conv_opj_final_name, index=True)
        fig4_title = " conv op - GOP/J compare  "
        conv_opj_pt = conv_opj_final_df.plot(kind="bar", title=fig4_title, logy=True, rot=0)
        conv_opj_fig = conv_opj_pt.get_figure()
        fig4_name = "results/bench/fig_mat_opj" + ".jpg"
        conv_opj_fig.savefig(fig4_name)        
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

    def bench_sample(self):
        func_map = self.make_hardware_func()

        with torch.no_grad():
            for batch_size in self.batch_size_list:
                benchmark_sample_s = {}
                benchmark_sample_j = {}

                for hardware_type in self.hardware_info.keys():
                    sample_s_record = []
                    sample_j_record = []

                    for model_name in self.model_list:
                        sample_s, sample_j= func_map[hardware_type](model_name, batch_size, "sample")
                        sample_s_record.append(sample_s)
                        sample_j_record.append(sample_j)

                    benchmark_sample_s[hardware_type] = sample_s_record
                    benchmark_sample_j[hardware_type] = sample_j_record
                
                hprefix = self.get_prefix()

                file_name_samples = "results/bench/samples_" + str(batch_size) + "_" + hprefix
                benchmark_ops_new = pandas.DataFrame(benchmark_sample_s)
                benchmark_ops_new.to_csv(file_name_samples)                   

                file_name_samplej = "results/bench/samplej_" + str(batch_size) + "_" + hprefix
                benchmark_opj_new = pandas.DataFrame(benchmark_sample_j)
                benchmark_opj_new.to_csv(file_name_samplej)
        
        return

    def bench_micro(self, mat_list, conv_list, device="cpu"):
        mat_ops = []
        mat_opj = []

        for i in range(len(mat_list)):
            model = pm.__dict__[mat_list[i]]()
            if mat_list[i] == "matmul256":
                img = torch.randn(1, 1, 256, 256)
            elif mat_list[i] == "matmul1024":
                img = torch.randn(1, 1, 1024, 1024)
            elif mat_list[i] == "matmul4096":
                img = torch.randn(1, 1, 4096, 4096)
            
            if device == "gpu":
                model = model.to("cuda")
                img = img.to("cuda")
            elif device == "npu":
                model = model.to("npu:0")
                img = img.to("npu:0")

            op_num = sop_dir[mat_list[i]]
            sum_time = 0
            
            for j in range(self.warm_up+self.infer_epoch):
                if j < self.warm_up:
                    outputs = model(img)
                else:
                    if device == "cpu":
                        op_time = self.micro_cpu()
                    elif device == "gpu":
                        op_time = self.micro_gpu()
                    elif device == "npu":
                        op_time =  self.micro_npu()
                    sum_time += op_time
                
            ops = (op_num * self.infer_epoch) / sum_time
            opj = ops / self.hardware_info[device]

            mat_ops.append(ops)
            mat_opj.append(opj)

        bench_mat_ops = {}
        bench_mat_ops[device] = mat_ops
        bench_mat_opj = {}
        bench_mat_opj[device] = mat_opj

        file_mat_ops = "results/bench/mat_ops_" + device 
        bench_mat_ops_new = pandas.DataFrame(bench_mat_ops)
        bench_mat_ops_new.to_csv(file_mat_ops)

        file_mat_opj = "results/bench/mat_opj_" + device 
        bench_mat_opj_new = pandas.DataFrame(bench_mat_opj)
        bench_mat_opj_new.to_csv(file_mat_opj)

        conv_ops = []
        conv_opj = []

        for i in range(len(conv_list)):
            model = pm.__dict__[conv_list[i]]()
            if conv_list[i] == "cnsmall":
                img = torch.randn(64, 3, 224, 224)
            elif conv_list[i] == "cnmid":
                img = torch.randn(128, 3, 224, 224)
            elif conv_list[i] == "cnbig":
                img = torch.randn(256, 3, 224, 224)
            
            if device == "gpu":
                model = model.to("cuda")
                img = img.to("cuda")
            elif device == "npu":
                model = model.to("npu:0")
                img = img.to("npu:0")

            op_num = sop_dir[conv_list[i]]
            sum_time = 0
            
            for j in range(self.warm_up+self.infer_epoch):
                if j < self.warm_up:
                    outputs = model(img)
                else:
                    if device == "cpu":
                        op_time = self.micro_cpu()
                    elif device == "gpu":
                        op_time = self.micro_gpu()
                    elif device == "npu":
                        op_time =  self.micro_npu()
                    sum_time += op_time / 1000
                
            ops = sum_time / (op_num * self.infer_epoch)
            opj = ops / self.hardware_info[device]

            conv_ops.append(ops)
            conv_opj.append(opj)

        bench_conv_ops = {}
        bench_conv_ops[device] = conv_ops
        bench_conv_opj = {}
        bench_conv_opj[device] = conv_opj

        file_conv_ops = "results/bench/conv_ops_" + device 
        bench_conv_ops_new = pandas.DataFrame(bench_conv_ops)
        bench_conv_ops_new.to_csv(file_conv_ops)

        file_conv_opj = "results/bench/conv_opj_" + device 
        bench_conv_opj_new = pandas.DataFrame(bench_conv_opj)
        bench_conv_opj_new.to_csv(file_conv_opj)
          
        return


if __name__ == "__main__":
    warm_up = 5
    infer_epoch = 5
    batch_size_list = [1, 16, 64]

    model_list = ["senet154", "se_resnext50_32x4d", "efficientnet_b3", "unet", "unetpp",
            "mgn", "osnet", "pcb", "baseline", "alphapose", "st_gcn_net", "deeppose"]
    model_simple = {0:"se154", 1:"se50", 2:"eb3", 3:"unet", 4:"unet++", 
            5:"mgn", 6:"osnet", 7:"pcb", 8:"bline", 9:"apose", 10:"stgcn", 11:"dpose"}
    
    hardware_info = {"gpu":250}
    
    dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
                            model_list, model_simple, hardware_info)
    dlh_bench.bench_opsj()

    """
    dlh_bench.bench_sample()

    warm_up = 5
    infer_epoch = 5
    batch_size_list = [1, 16, 64]

    model_list = ["senet154", "se_resnext50_32x4d", "unet", "unetpp",
            "mgn", "baseline", "deeppose"]
    model_simple = {0:"se154", 1:"se50", 2:"unet", 3:"unet++", 
            4:"mgn", 5:"bline", 6:"dpose"}
    
    hardware_info = {"gpu":250}
    
    dlh_bench = DLHBenchmark(warm_up, infer_epoch, batch_size_list,
                            model_list, model_simple, hardware_info)
    dlh_bench.bench_opsj()
    dlh_bench.bench_sample()
    """

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
