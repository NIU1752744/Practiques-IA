__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils import *
from KNN import *
from utils_data import *


if __name__ == '__main__':

    

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    knn = KNN(train_imgs, train_class_labels)
    #knn.get_k_neighbours(test_color_labels, 5)
    knn.get_k_neighbours(test_imgs, 5)

    print(knn.get_class())
    #test =knn._init_train(train_imgs)
    #print(len(knn.train_data))

    #print(len(test_class_labels))
    





    from utils import *
    import numpy as np
    from PIL import Image
    from Kmeans import KMeans
    from Kmeans import get_colors, distance

    Path_to_img = './images/4solid_colors.jpg'
    img = Image.open(Path_to_img)
    img = img.convert('RGB')
    km = KMeans(img, K=4, options={"km_init": "random"})
    km.fit()
    print(km.centroids)
    
    print(get_colors(km.centroids))


    """
    import pickle
    from utils import *
    from Kmeans import *

    np.random.seed(666)
    with open('./test/test_cases_kmeans.pkl', 'rb') as f:
        test_cases = pickle.load(f)
    for ix, input in enumerate(test_cases['input']):
                km = KMeans(input, test_cases['K'][ix])
                km.find_bestK(10)
                print(km.K)
    """