import os
import random
import math

ITER = 2000
i = 666
j = 0

current_dir = os.getcwd()
folder_shapes = os.path.join(current_dir,"shapes")
files = os.listdir(folder_shapes)

for f in files:

    if("input_{f[6:]}-%3.1f.gfs" in files == False):

        file = open("final.gfs")
        data = file.readlines()

        data[4] = f'\tGfsSolid {f}\n'
        data[31] = f'\tGfsOutputSimulation {{start = start}} input_{f[6:]}-%3.1f.gfs {{\n'
        data[35] = f'\tGfsOutputSimulation {{start = end}} output_{f[6:]}-%3.1f.gfs {{\n'

        data[16] = f'\tOutputPPM {{start = end}} {{ convert -colors 256 ppm:- velocity_{f[6:]}.png}} {{\n'
        data[19] = f'\tOutputPPM {{start = end}} {{ convert -colors 256 ppm:- velocity_{f[6:]}.png}} {{\n'

        with open("final.gfs","w") as k:
            k.writelines(data)
            k.close()

        try:
            os.system(f"gerris2D final.gfs")
            # # all_gfs.append(i)
            # print(f"Done with file {i}")
            # print("Animating.....")
            # os.system(f"animate T_final.ppm")
        except:
            print(f"Error in file {i}")

        j+=1

        if(j>=1):
            break