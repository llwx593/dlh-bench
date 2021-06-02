import pandas

pic_name = ["conv_ops"]

# model_simple = {0:"se154", 1:"se50", 2:"eb3", 3:"unet", 4:"unet++", 
#            5:"mgn", 6:"osnet", 7:"pcb", 8:"bline", 9:"apose", 10:"stgcn", 11:"dpose"}

# model_simple = {0:"se154", 1:"se50", 2:"unet", 3:"unet++", 
#             4:"mgn", 5:"bline"}

model_simple = {0:"Conv2d I", 1:"Conv2d II", 2:"Conv2d III"}

# model_simple = {0:"Matmul I", 1:"Matmul II", 2:"Matmul III"}

num_list = [1, 16, 64]
for i in range(len(pic_name)):
    file_name_ops = "results/final/" + pic_name[i]
    csvdata = pandas.read_csv(file_name_ops)
    df = pandas.DataFrame(csvdata)
    df_new = df.rename(index = model_simple)
    #fig_title = "GFLOPS Compare" + " Batch Size = " + str(num_list[i])
    samples_pt = df_new.plot(ylabel='GFLOPS', kind="bar", logy=True, rot=0, fontsize=12)
    samples_fig = samples_pt.get_figure()
    fig_name = "results/final/" + pic_name[i] + ".jpg"
    samples_fig.savefig(fig_name)