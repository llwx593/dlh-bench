import pretrained_models as pm
import torch
import torch.nn as nn
import time

model_name = ["cnsmall", "cnmid", "cnbig", "matmul256", "matmul1024", "matmul4096"]
op_dir = [0.04, 0.36, 0.7, 0.0336, 2.15, 137.0]
pow_dir = [150, 250, 260]
device = "cpu"
warm_up = 5
loop_time = 5
ops_list = []
opj_list = []

def make_op_test():
    for i in range(len(model_name)):
        model = pm.__dict__[model_name[i]]()
        if i < 3:
            img = torch.randn(1,3,224,224)
        elif i == 3:
            img = torch.randn(1,1,256,256)
        elif i == 4:
            img = torch.randn(1,1,1024,1024)
        elif i == 5:
            img = torch.randn(1,1,4096,4096)

        if device == "gpu":
            model = model.cuda()
            img = img.cuda()
        elif device == "npu":
            model = model.npu()
            img = img.npu()

        sum = 0
        for j in range(warm_up + loop_time):
            if j < warm_up:
                outputs = model(img)
            else:
                start_time = time.time()
                outputs = model(img)
                end_time = time.time()
                if device == "gpu":
                    torch.cuda.synchronize()
                elif device == "npu":
                    torch.npu.synchronize()
                durations = end_time - start_time
                sum += durations
        sum = sum / loop_time*1.0
        ops = op_dir[i] / sum
        pow_num = ""
        if device == "cpu":
            pow_num = pow_dir[0]
        elif device == "gpu":
            pow_num = pow_dir[1]
        elif device == "npu":
            pow_num = pow_dir[2]
        opj = ops / pow_num
        ops_list.append(ops)
        opj_list.append(opj)
    
    return ops_list, opj_list


def show_ans(ops, opj):
    print("======ops list======")
    print(ops)

    print("======opj list======")
    print(opj)
    return

if __name__ == "__main__":
    ops, opj = make_op_test()
    show_ans(ops, opj)