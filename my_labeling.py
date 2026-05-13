__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils import *
from KNN import *
from Kmeans import *
from utils_data import *


def Retrieval_by_color(imgs, color, n_items):
    selected_imgs = []
    selected_labels = []
    n = 0
    i = 0
    while n < n_items and i < len(imgs):
        img = imgs[i]
        km = KMeans(img, K=3, options={"km_init": "random"})
        km.fit()
        colors = get_colors(km.centroids)
        if color in colors:
            selected_imgs.append(imgs[i])
            selected_labels.append((str(colors[0]), str(colors[1]), str(colors[2])))
            n += 1
        i += 1
    return selected_imgs, selected_labels


def Retrieval_by_shape(imgs, labels, shape, n_items):
    indexes = []
    n = 0
    i = 0
    selected_imgs = []
    selected_labels = []
    while n < n_items and i < len(labels):
        if labels[i] == shape:
            selected_imgs.append(imgs[i])
            selected_labels.append(labels[i])            
            n += 1
        i += 1
    return selected_imgs, selected_labels


def Retrieval_combined(imgs, labels, shape, n_items, color):
    selected_imgs, selected_labels = Retrieval_by_shape(imgs, labels, shape, n_items)
    selected_imgs, selected_labels = Retrieval_by_color(selected_imgs, color, n_items)
    return selected_imgs, selected_labels
    

if __name__ == '__main__':

    

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)
    
    train_imgs_color, train_class_labels_color, train_color_labels_color, test_imgs_color, test_class_labels_color, \
        test_color_labels_color = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=True)

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    option = -1

    while option != 0:
        print("Welcome to the Clothes Finder Program! Please select a function:\n")
        print("1. Retrieval by color")
        print("2. Retrieval by shape")
        print("3. Combined retrieval (color + shape)")
        print("0. Exit")

        function = int(input("Enter a number (0-3): "))
        if function == 0:
            break
        elif function == 1:
            print("Which color do you want to find?\n1. Red\n2. Orange\n3. Brown\n4. Yellow\n5. Green\n6. Blue\n7. Purple\n8. Pink\n9. Black\n10. Grey\n11. White")

            choice = int(input("Enter a number (1-11): "))

            if choice == 1:
                color = "Red"
            elif choice == 2:
                color = "Orange"
            elif choice == 3:
                color = "Brown"
            elif choice == 4:
                color = "Yellow"
            elif choice == 5:
                color = "Green"
            elif choice == 6:
                color = "Blue"
            elif choice == 7:
                color = "Purple"
            elif choice == 8:
                color = "Pink"
            elif choice == 9:
                color = "Black"
            elif choice == 10:
                color = "Grey"
            elif choice == 11:
                color = "White"
            else:
                color = "Invalid option"
            n_items = int(input("How many images do you want to search? Enter a number: "))
            print("Please wait...")
            selected_imgs, selected_labels = Retrieval_by_color(test_imgs_color, color, n_items)
            visualize_retrieval(selected_imgs, n_items, info=selected_labels, ok=None, title='', query=None)

        elif function == 2:
            print("Which shape do you want to find?\n1. Dresses\n2. Flip Flops\n3. Jeans\n4. Sandals\n5. Shirts\n6. Shorts\n7. Socks\n8. Handbags")
            choice = int(input("Enter a number (1-8): "))

            if choice == 1:
                shape = "Dresses"
            elif choice == 2:
                shape = "Flip Flops"
            elif choice == 3:
                shape = "Jeans"
            elif choice == 4:
                shape = "Sandals"
            elif choice == 5:
                shape = "Shirts"
            elif choice == 6:
                shape = "Shorts"
            elif choice == 7:
                shape = "Socks"
            elif choice == 8:
                shape = "Handbags"
            else:
                shape = "Invalid option"
            n_items = int(input("How many images do you want to search? Enter a number: "))
            print("Please wait...")
            knn = KNN(train_imgs, train_class_labels)
            labels = knn.predict(test_imgs, 5)

            selected_imgs, selected_labels = Retrieval_by_shape(test_imgs_color, labels, shape, n_items)
            visualize_retrieval(selected_imgs, n_items, info=selected_labels, ok=None, title='', query=None)

        elif function == 3:
            print("Which color do you want to find?\n1. Red\n2. Orange\n3. Brown\n4. Yellow\n5. Green\n6. Blue\n7. Purple\n8. Pink\n9. Black\n10. Grey\n11. White")

            choice = int(input("Enter a number (1-11): "))

            if choice == 1:
                color = "Red"
            elif choice == 2:
                color = "Orange"
            elif choice == 3:
                color = "Brown"
            elif choice == 4:
                color = "Yellow"
            elif choice == 5:
                color = "Green"
            elif choice == 6:
                color = "Blue"
            elif choice == 7:
                color = "Purple"
            elif choice == 8:
                color = "Pink"
            elif choice == 9:
                color = "Black"
            elif choice == 10:
                color = "Grey"
            elif choice == 11:
                color = "White"
            else:
                color = "Invalid option"
            
            print("Which shape do you want to find?\n1. Dresses\n2. Flip Flops\n3. Jeans\n4. Sandals\n5. Shirts\n6. Shorts\n7. Socks\n8. Handbags")
            choice = int(input("Enter a number (1-8): "))

            if choice == 1:
                shape = "Dresses"
            elif choice == 2:
                shape = "Flip Flops"
            elif choice == 3:
                shape = "Jeans"
            elif choice == 4:
                shape = "Sandals"
            elif choice == 5:
                shape = "Shirts"
            elif choice == 6:
                shape = "Shorts"
            elif choice == 7:
                shape = "Socks"
            elif choice == 8:
                shape = "Handbags"
            else:
                shape = "Invalid option"
            n_items = int(input("How many images do you want to search? Enter a number: "))
            print("Please wait...")
            knn = KNN(train_imgs, train_class_labels)
            labels = knn.predict(test_imgs, 5)
            selected_imgs, selected_labels = Retrieval_combined(test_imgs_color, labels, shape, n_items, color)
            visualize_retrieval(selected_imgs, n_items, info=selected_labels, ok=None, title='', query=None)

            



    """

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