import pretrained_models as pm
import torch
import torch.nn as nn
import time

model_list = ["senet154", "se_resnext50_32x4d", "efficientnet_b3", "unet", "unetpp",
        "mgn", "osnet", "pcb", "baseline", "alphapose", "st_gcn_net", "deeppose"]

#model_list = ["senet154", "se_resnext50_32x4d", "unet", "unetpp",
#        "mgn", "baseline", "deeppose"]

op_dir = {"senet154": 41.64, "se_resnext50_32x4d": 8.56, "efficientnet_b3": 1.8, "unet": 61.42, "unetpp": 53.06, "mgn": 23.88, "osnet": 
0.98, "pcb": 8.08, "baseline": 8.08, "alphapose": 11.82, "st_gcn_net": 7.8, "deeppose": 8.24}

batch_size = 1

power_num = 250

ops_list = {}
opj_list = {}

for i in range(len(model_list)):
    model_name = model_list[i]
    img = torch.randn(batch_size, 3, 224, 224)
    img_warm_up = torch.randn(1, 3, 224, 224)
    if model_name == "osnet":
        img = torch.randn(batch_size, 3, 256, 128)
        img_warm_up = torch.randn(1, 3, 256, 128)
    elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
        img = torch.randn(batch_size, 3, 384, 128)
        img_warm_up = torch.randn(1, 3, 384, 128)
    elif model_name == "alphapose":
        img = torch.randn(batch_size, 3, 256, 192)
        img_warm_up = torch.randn(1, 3, 256, 192)
    elif model_name == "st_gcn_net":
        img = torch.randn(batch_size, 3, 256, 14)
        img_warm_up = torch.randn(1, 3, 256, 14)
    model = pm.__dict__[model_name]()
    model = model.to("cuda:1")
    model.eval()
    img = img.to("cuda:1")
    img_warm_up = img_warm_up.to("cuda:1")
    with torch.no_grad():
        for i in range(5):
            outputs = model(img_warm_up)
        start_time = time.time()
        outputs = model(img)
        end_time = time.time()
        torch.cuda.synchronize()
        durations = (end_time - start_time) * 1000
        print("-------")
        print(durations)
        print("now model is ", model_name)
        run_time = durations / 1000
        now_ops = (op_dir[model_name] * batch_size) / run_time
        now_opj = now_ops / power_num
        ops_list[model_name] = now_ops
        opj_list[model_name] = now_opj

print("======")
print(ops_list)
print(opj_list)


# model = pm.__dict__["unet"]()
# #model = model.to("npu:0")
# img = torch.randn(1, 3, 224, 224) 
# #img = img.to("npu:0")

# with torch.no_grad():
#     start_time = time.time()
#     outputs = model(img)
#     end_time = time.time()
#     durations = (end_time - start_time) * 1000
#     print(durations)

