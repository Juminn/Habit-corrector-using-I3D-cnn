import os
from random import sample

images_root = "C:\project\I3D-Tensorflow-master\images"
label = {'nomove':'0', 'fingernail': '1'}
with open(".\\list\\train.txt", 'w') as tr, open(".\\list\\validation.txt", 'w') as  val:
    for (path, dir, files) in os.walk(images_root):
        check = path.split('\\')[-1] #마지막폴더이름
        if check == 'fingernail' or check == 'nomove':
            dir_len = len(dir)
            print(dir_len)
            val_index = sample(range(0, dir_len), dir_len // 4)
            val_index = sorted(val_index)
            val_index.reverse()
            print(val_index)
            for i in val_index:
                tmp = os.path.join(path, dir.pop(i)) + ' ' + label[check] + '\n'
                val.write(tmp)
               # if i != val_index[-1]:
                #    val.write('\n')
            for i in dir:
                tmp = os.path.join(path, i) + ' ' + label[check] + '\n'
                tr.write(tmp)

             #   if i != dir[-1]:
              #      tr.write('\n')


print("done.")
