import pretrained_models as pm
from ptflops import get_model_complexity_info


# model_list = ["senet154", "se_resnext50_32x4d", "efficientnet_b3", "unet", "unetpp",
#         "mgn", "osnet", "pcb", "baseline", "alphapose", "st_gcn_net", "deeppose"]


model_list = ["matmul256", "matmul1024", "matmul4096", "cnsmall", "cnmid", "cnbig"]

def transStr2Float(input_str):
    for c in range(len(input_str)):
        if input_str[c] == 'G':
            return float(input_str[:c])


op_dir = {}
for i in range(len(model_list)):
    model_name = model_list[i]
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
    elif model_name == "matmul256":
        input_res = (1, 256, 256)
    elif model_name == "matmul1024":
        input_res = (1, 1024, 1024)
    elif model_name == "matmul4096":
        input_res = (1, 4096, 4096)

    # 获得模型的op
    macs, _ = get_model_complexity_info(model, input_res, as_strings=True, 
                                        print_per_layer_stat=False, verbose=True)                                                
    float_macs = transStr2Float(macs)
    op_num = float_macs * 2
    # 先使用paper给的值
    if model_name == "efficientnet_b3":
        op_num = 1.8
    elif model_name == "osnet":
        op_num = 0.98
    op_dir[model_name] = op_num

print(op_dir)
