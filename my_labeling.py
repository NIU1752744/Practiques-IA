__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils import *
from KNN import *
from Kmeans import *
from utils_data import *
import time


def Retrieval_by_color(imgs, color, n_items, option, llindar):
    indexes = []
    selected_imgs = []
    selected_labels = []
    statistics = []
    n = 0
    i = 0
    while n < n_items and i < len(imgs):
        img = imgs[i]
        if option == 1:
            km = KMeans(img, K=2, options={"km_init": "random"})
        else:
            km = KMeans(img, K=1, options={"km_init": "random"})
            km.llindar = llindar
            km.find_bestK(max_K=5)
        km.fit()
        #Plot3DCloud(km)
        colors = get_colors(km.centroids)
        if len(color) == 1:
            if color[0] in colors:
                selected_imgs.append(imgs[i])
                if option == 1:
                    selected_labels.append((str(colors[0]), str(colors[1])))
                else:
                    selected_labels.append(tuple(str(c) for c in colors))
                indexes.append(i)
                n += 1
            i += 1
        else:
            if (color[0] in colors) and (color[1] in colors):
                selected_imgs.append(imgs[i])
                if option == 1:
                    selected_labels.append((str(colors[0]), str(colors[1]), str(colors[2])))
                else:
                    selected_labels.append(tuple(str(c) for c in colors))
                indexes.append(i)
                n += 1
            i += 1
    return np.array(selected_imgs), selected_labels, indexes, statistics


def Retrieval_by_shape(imgs, labels, shape, n_items):
    indexes = []
    n = 0
    i = 0
    selected_imgs = []
    selected_labels = []
    correct = []
    while n < n_items and i < len(labels):
        if labels[i] == shape:
            selected_imgs.append(imgs[i])
            selected_labels.append(labels[i])         
            indexes.append(i) 
            n += 1
        i += 1
    return np.array(selected_imgs), selected_labels, indexes


def Retrieval_combined(imgs, labels, shape, n_items, color):
    #selected_imgs, selected_labels = Retrieval_by_shape(imgs, labels, shape, n_items)
    #selected_imgs, selected_labels = Retrieval_by_color(selected_imgs, color, n_items)
    indexes = []
    selected_imgs = []
    selected_labels = []
    n = 0
    i = 0
    while n < n_items and i < len(labels):
        if labels[i] == shape:
            km = KMeans(imgs[i], K=2, options={"km_init": "random"})
            km.fit()
            colors = get_colors(km.centroids)
            if color in colors:
                selected_imgs.append(imgs[i])
                selected_labels.append(tuple(str(c) for c in colors))
                indexes.append(i)
                n+=1
        i+=1

    return np.array(selected_imgs), selected_labels, indexes

def Get_shape_accuracy(correct_shapes):
    return correct_shapes.tolist().count(True) / len(correct_shapes)

def Get_color_accuracy(imgs, test_labels, option, llindar):
    labels = []
    for i in range(0, len(imgs)):
        if option == 1:
            km = KMeans(imgs[i], K=2, options={"km_init": "random"})
        else:
            km = KMeans(imgs[i], K=1, options={"km_init": "random"})
            km.llindar = llindar
            km.find_bestK(max_K=5)
        km.fit()
        colors = get_colors(km.centroids)
        labels.append(colors)
    correct_predictions = 0
    incorrect_predictions = 0
    for i in range(0, len(labels)):
        correct = 0
        incorrect = 0
        for j in labels[i]:
            if j in test_labels[i]:
                correct += 1
                
            else:
                incorrect += 1
        if correct >= incorrect:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
    #print(correct_predictions + incorrect_predictions)
    return correct_predictions / (correct_predictions + incorrect_predictions)
def Kmean_statistics(images, Kmax=10):
    """
    images: llista d'imatges
    """
    n_imgs = len(images)
    wcd_values = []
    iterations = []
    times = []
    intra_means = []
    inter_means = []
    fisher_means = []
    K_range = range(2, Kmax + 1)

    plt.figure(figsize=(15, 5))
    plt.suptitle("3D Pixel Clouds for the First Image across K")


    for k_idx, k in enumerate(K_range):
        wcd_k, iter_k, time_k = [], [], []
        intra_k, inter_k, fisher_k = [], [], []
        for img_idx, img in enumerate(images):
            start = time.time()
            km = KMeans(img, K=k, options={'km_init': 'random', 'max_iter': 100, 'tolerance': 0})
            km.fit()
            end = time.time()
            wcd_k.append(km.withinClassDistance())
            iter_k.append(km.num_iter)
            time_k.append(end - start)

            # distancies intra i inter i el seu ratio
            labels = km.labels
            X = km.X
            intra = intra_class_distance(X, labels)
            inter = inter_class_distance(X, labels)
            intra_k.append(intra)
            inter_k.append(inter)
            fisher_k.append(inter / intra if intra != 0 else 0)
        
            if img_idx == 4:
                ax = Plot3DCloud(km, rows=1, cols=len(K_range), spl_id=k_idx+1)
                plt.title(f"K = {k}")

        wcd_values.append(np.mean(wcd_k))
        iterations.append(np.mean(iter_k))
        times.append(np.mean(time_k))
        intra_means.append(np.mean(intra_k))
        inter_means.append(np.mean(inter_k))
        fisher_means.append(np.mean(fisher_k))
        print(f"K={k} | "
              f"WCD(mean)={wcd_values[-1]:.2f} | "
              f"intra(mean)={intra_means[-1]:.2f} | "
              f"inter(mean)={inter_means[-1]:.2f} | "
              f"Fisher(mean)={fisher_means[-1]:.2f} | "
              f"iters(mean)={iterations[-1]:.2f} | "
              f"time(mean)={times[-1]:.4f}s")

    #grafic wcd
    plt.figure()
    plt.plot(K_range, wcd_values, marker='o', label="WCD")
    plt.xlabel("K")
    plt.ylabel("WCD")
    plt.title("KMeans Average Within-Class Distance (WCD)")
    plt.legend()

    # grafic intra/inter/fisher
    plt.figure()
    plt.plot(K_range, intra_means, marker='o', label="Intra-class")
    plt.plot(K_range, inter_means, marker='o', label="Inter-class")
    plt.plot(K_range, fisher_means, marker='o', label="Fisher")
    plt.xlabel("K")
    plt.ylabel("Value")
    plt.title("KMeans Intra-class, Inter-class & Fisher")
    plt.legend()

    #grafic iterations
    plt.figure()
    plt.plot(K_range, iterations, marker='o')
    plt.xlabel("K")
    plt.ylabel("Mean Iterations")
    plt.title("Iterations until Convergence")

    #grafic temps
    plt.figure()
    plt.plot(K_range, times, marker='o')
    plt.xlabel("K")
    plt.ylabel("Mean Time (s)")
    plt.title("Execution Time for KMeans")
    plt.show()

    


def inter_class_distance(X, labels):
    """
    Calcula la distancia media entre todos los pares de centroides (inter-clase).
    X: (N, D) array con datos.
    labels: (N,) array con la etiqueta (cluster) de cada punto.
    Devuelve: valor escalar de la distancia media inter-clase.
    """
    unique_labels = np.unique(labels)
    centroids = []
    for cl in unique_labels:
        class_points = X[labels == cl]
        if len(class_points) == 0:
            continue
        centroids.append(np.mean(class_points, axis=0))
    centroids = np.array(centroids)
    if len(centroids) < 2:
        return 0.0

    total_dist = 0.0
    count = 0
    n = len(centroids)
    for i in range(n):
        for j in range(i+1, n):
            total_dist += np.linalg.norm(centroids[i] - centroids[j])
            count += 1
    if count == 0:
        return 0.0
    return total_dist / count


def intra_class_distance(X, labels):
    unique_labels = np.unique(labels)
    all_distances = []
    for cl in unique_labels:
        class_points = X[labels == cl]
        if len(class_points) == 0: continue
        centroid = np.mean(class_points, axis=0)
        dist = np.linalg.norm(class_points - centroid, axis=1)
        all_distances.extend(dist)
    return np.mean(all_distances)

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
    knn = KNN(train_imgs, train_class_labels)
    labels = knn.predict(test_imgs, 5)
    
    correct_shapes = labels == test_class_labels

    while option != 0:
        print("Welcome to the Clothes Finder Program! Please select a function:\n")
        print("1. Retrieval by color")
        print("2. Retrieval by shape")
        print("3. Combined retrieval (color + shape)")
        print("4. Get shape accuracy")
        print("5. Get color accuracy")
        print("6. Test KNN for different K values")
        print("7. Run Kmeans statistics")
        print("8. Test different threshold values")
        print("0. Exit")

        function = int(input("Enter a number (0-8): "))
        if function == 0:
            break
        elif function == 1:
            print("Which color do you want to find?\n1. Red\n2. Orange\n3. Brown\n4. Yellow\n5. Green\n6. Blue\n7. Purple\n8. Pink\n9. Black\n10. Grey\n11. White")

            choice = int(input("Enter a number (1-11): "))
            print("Note: searching for more than 2 colors is not recommended due to innacurate results")
            choice2 = int(input("Enter 0 to continue or enter another number to search for 2 colors: "))
            colors = []

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
            colors.append(color)
            if choice2 != 0:
                if choice2 == 1:
                    color2 = "Red"
                elif choice2 == 2:
                    color2 = "Orange"
                elif choice2 == 3:
                    color2 = "Brown"
                elif choice2 == 4:
                    color2 = "Yellow"
                elif choice2 == 5:
                    color2 = "Green"
                elif choice2 == 6:
                    color2 = "Blue"
                elif choice2 == 7:
                    color2 = "Purple"
                elif choice2 == 8:
                    color2 = "Pink"
                elif choice2 == 9:
                    color2 = "Black"
                elif choice2 == 10:
                    color2 = "Grey"
                elif choice2 == 11:
                    color2 = "White"
                else:
                    color2 = "Invalid option"
                colors.append(color2)
            n_items = int(input("How many images do you want to search? Enter a number: "))
            print("Choose the value to assign to K:\n1. K=2 always (faster)\n2. Run find_bestK for each image (slower)")
            option = int(input("Enter a number (1 or 2): "))
            threshold = -1
            if option == 2:
                threshold = int(input("Choose threshold: "))  
            print("Please wait...")
            selected_imgs, selected_labels, selected_indexes, statistics = Retrieval_by_color(test_imgs_color, colors, n_items, option, threshold)
            test_color_labels_reduced = test_color_labels[selected_indexes]
            correct_colors = []
            if choice2 == 0:
                for i in test_color_labels_reduced:
                    if color in i:
                        correct_colors.append(True)
                    else:
                        correct_colors.append(False)
            else:
                for i in test_color_labels_reduced:
                    if (colors[0] in i) and (colors[1] in i):
                        correct_colors.append(True)
                    else:
                        correct_colors.append(False)
            accuracy = correct_colors.count(True)/len(correct_colors)
            visualize_retrieval(selected_imgs, n_items, info=test_color_labels_reduced, ok=correct_colors, title=("Searched for",colors,"Accuracy:",accuracy), query=None)
            print("Accuracy:", accuracy)

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

            selected_imgs, selected_labels, selected_indexes = Retrieval_by_shape(test_imgs_color, labels, shape, n_items)
            correct = correct_shapes[selected_indexes]
            #visualize_retrieval(selected_imgs, n_items, info=selected_labels, ok=correct, title='', query=None)
            accuracy = correct.tolist().count(True)/len(correct)
            visualize_retrieval(selected_imgs, n_items, info=test_class_labels[selected_indexes], ok=correct, title=("Searched for",shape,"Accuracy:",accuracy), query=None)
            print("Accuracy:", accuracy)



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
            selected_imgs, selected_labels, selected_indexes = Retrieval_combined(test_imgs_color, labels, shape, n_items, color)
            correct_shapes_reduced = correct_shapes[selected_indexes]
            test_color_labels_reduced = test_color_labels[selected_indexes]
            correct_colors = []
            for i in test_color_labels_reduced:
                if color in i:
                    correct_colors.append(True)
                else:
                    correct_colors.append(False)
            correct_combined = []
            for i in range(0, len(correct_colors)):
                if correct_colors[i] == correct_shapes_reduced[i] == True:
                    correct_combined.append(True)
                else:
                    correct_combined.append(False)
            accuracy = correct_combined.count(True)/len(correct_combined)
            visualize_retrieval(selected_imgs, n_items, info=test_color_labels_reduced, ok=correct_combined, title=("Searched for",color, shape,"Accuracy:",accuracy), query=None)
            print("Accuracy:", accuracy)
        elif function == 4:
            print(Get_shape_accuracy(correct_shapes))
        elif function == 5:
            print("Choose the value to assign to K:\n1. K=2 always (faster)\n2. Run find_bestK for each image (slower)")
            option = int(input("Enter a number (1 or 2): "))
            threshold = -1
            if option == 2:
                threshold = int(input("Choose threshold: "))  
            print(Get_color_accuracy(test_imgs_color, test_color_labels, option, threshold))
        elif function == 6:
            time_list = []
            accuracy_list = []

            for i in range(1, 6):
                time_start = time.time()
                labels = knn.predict(test_imgs, i)
                time_end = time.time()
                correct_shapes = labels == test_class_labels
                accuracy_list.append(Get_shape_accuracy(correct_shapes))
                time_list.append(time_end - time_start)
            plt.figure()
            plt.plot(range(1, 6), time_list)
            plt.xticks(range(1, 6))
            plt.xlabel("K")
            plt.ylabel("Time")
            plt.title("KNN time for each K")

            plt.figure()
            plt.plot(range(1, 6), accuracy_list)
            plt.xticks(range(1, 6))
            plt.xlabel("K")
            plt.ylabel("Accuracy")
            plt.title("KNN accuracy for each K")

            plt.show()
        elif function == 7:
            n_images = int(input("Enter number of images to use: "))
            Kmean_statistics(test_imgs_color[0:n_images], Kmax=5)
        elif function == 8:
            threshold_start = 0
            threshold_end = 100
            n_images = int(input("Enter number of images to use: "))
            accuracies = []
            thresholds = []
            for i in range(threshold_start, threshold_end + 1):
                print("Computing accuracy for threshold",i)
                accuracies.append(Get_color_accuracy(test_imgs_color[0:n_images], test_color_labels[0:n_images], 2, i))
                thresholds.append(i)
            
            plt.figure()
            plt.plot(thresholds, accuracies)
            #plt.xticks(llindars)
            plt.xlabel("Valor llindar (per defecte és 20%)")
            plt.ylabel("Accuracy")
            plt.title("Precisió de Kmeans per diferents valors del llindar de WCD")
            plt.show()
