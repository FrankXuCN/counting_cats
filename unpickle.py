
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

if __name__ == '__main__':
    import os
    import sys
    from PIL import Image
    import matplotlib.pyplot as plt
    import scipy.misc as smp

    file = sys.argv[1]
    dict_obj = unpickle(file)
    for item in dict_obj:
        print(item)

    batch_label = b'batch_label'
    if batch_label in dict_obj:
        batch_labels = dict_obj.get(batch_label)
        # if len(batch_labels) < 10000:
        print("1. {}".format(batch_labels))
    else:
        print("no {}".format(batch_label))

    t_label = b'labels'
    if t_label in dict_obj:
        labels = dict_obj.get(t_label)
        condition = 9
        if condition in [a for a in labels if a == condition]:
            print("get value is {}".format(condition))
        else:
            print("no value is {}".format(condition))

    tag = b'data'
    if tag in dict_obj and "-c" == sys.argv[2]:
        data = dict_obj.get(tag)
        print(len(data))
        basePath = "./data"
        folder = sys.argv[3]
        if folder not in os.listdir(basePath):
            basePath = os.path.join(basePath,folder)
            os.mkdir(basePath)
            print("create folder {}".format(basePath))
        else:
            basePath = os.path.join(basePath,folder)


        if len(data) == 10000:
            x = data.reshape(len(data),3,32,32).transpose(0, 2, 3, 1)
            for i in range(len(x)):
                img = Image.fromarray(x[i],'RGB')
                # img = smp.toimage(x[i])
                # img.show()
                savePath = os.path.join(basePath,"{}_{}.jpg".format(labels[i],i))
                print(savePath)
                img.save(savePath)
            print("over")

    tag = b'label_names'
    if tag in dict_obj:
        label_names = dict_obj.get(tag)
        print(label_names)
