import json

import numpy as np

def read_sloth(file):
    with open(file, 'r') as fp:
        sloth_data = json.load(fp)
    return sloth_data

def filter_annotations(annotations, annotation_class):
    return [ann for ann in annotations if ann['class'] == annotation_class]

def sloth_poly_to_array(poly,flip_order=False):
    xn = poly['xn'].split(';')
    xn = np.asarray([float(x) for x in xn]).reshape(-1,1)

    yn = poly['yn'].split(';')
    yn = np.asarray([float(y) for y in yn]).reshape(-1,1)

    if flip_order:
        return np.concatenate([xn,yn],axis=1)
    else:
        return np.concatenate([yn, xn], axis=1)

def iter_images(sloth_data):

    for item in sloth_data:
        if item['class'] =='image':
            if 'annotations' in item:
                yield item['filename'], item['annotations']
