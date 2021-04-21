import pretrained_models as pm
import torch
import torch.nn as nn
import time

# model_list = ["senet154", "se_resnext50_32x4d", "efficientnet_b3", "unet", "unetpp",
#         "mgn", "osnet", "pcb", "baseline", "alphapose", "st_gcn_net", "deeppose"]

# for i in range(len(model_list)):
#     model_name = model_list[i]
#     img = torch.randn(1, 3, 224, 224)
#     if model_name == "osnet":
#         img = torch.randn(1, 3, 256, 128)
#     elif model_name == "mgn" or model_name == "pcb" or model_name == "baseline":
#         img = torch.randn(1, 3, 384, 128)
#     elif model_name == "alphapose":
#         img = torch.randn(1, 3, 256, 192)
#     elif model_name == "st_gcn_net":
#         img = torch.randn(1, 3, 256, 14)
#     model = pm.__dict__[model_name]()
#     model = model.to("npu:0")
#     model.eval()
#     img = img.to("npu:0")
#     with torch.no_grad():
#         # for i in range(5):
#         #     outputs = model(img)
#         start_time = time.time()
#         outputs = model(img)
#         end_time = time.time()
#         durations = (end_time - start_time) * 1000
#         print("-------")
#         print(durations)
#         print("now model is ", model_name)

model = pm.__dict__["unet"]()
#model = model.to("npu:0")
img = torch.randn(1, 3, 224, 224)
#img = img.to("npu:0")

with torch.no_grad():
    start_time = time.time()
    outputs = model(img)
    end_time = time.time()
    durations = (end_time - start_time) * 1000
    print(durations)

