### ALL PLOTTING AND VISUALIZATIONS

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.patches import Patch

def visualize_real_vs_fake(class_count: list) -> None:
    plt.pie(class_count, labels=["Real Videos", "Deep Fake Videos"], 
                        autopct='%.2f%%',
                        wedgeprops={'linewidth': 2.5, 'edgecolor': 'white'},
                        textprops={'size': 'large', 'fontweight': 'bold'})
    
    plt.title("Proportion of Real vs Deep Fake videos in the training dataset.", 
                        fontdict={'fontweight': 'bold'})
    plt.legend([f"Real Videos Count: {class_count[0]}", f"Deep Fake Videos Count: {class_count[1]}"],
                        bbox_to_anchor=(0.5, 0.05), 
                        bbox_transform=plt.gcf().transFigure, 
                        loc="lower center", 
                        prop={'weight':'bold'})
    
    plt.savefig("images/pie_chart_class_proportions.jpg")
    plt.show()


def print_face_features(faces, indices):
    print(faces[0][1].shape)
    print(faces[1][1].shape)
    print(faces.shape)
    print(indices)

def plot_tsne_with_images(tsne_results, face_regions, figsize=(4, 4), thumbnail_size=(64, 36)):
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(tsne_results.shape[0]):
        x, y = tsne_results[i, :]
        img = face_regions[i]
        img = cv2.resize(img, thumbnail_size)
        img_box = offsetbox.OffsetImage(img, zoom=1, cmap='gray')
        img_annotation = offsetbox.AnnotationBbox(img_box, (x, y), xycoords='data', frameon=False)
        ax.add_artist(img_annotation)

    ax.set_xlim(tsne_results[:, 0].min() - 10, tsne_results[:, 0].max() + 10)
    ax.set_ylim(tsne_results[:, 1].min() - 10, tsne_results[:, 1].max() + 10)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('t-SNE Plot of Video Features with Face Thumbnails')
    plt.show()

def plot_video(video: list, figsize: tuple, width: int, height: int) -> None:
    fig = plt.figure(figsize=figsize)
    for i in range(len(video[:(width*height)])):
        plt.subplot(width, height, i+1)
        plt.imshow(video[i])
    plt.show()

def plot_faces(faces: list, figsize: tuple, width: int, height: int) -> None:
    fig = plt.figure(figsize=figsize)
    num_faces = min(len(faces), width * height)
    for i in range(num_faces):
        plt.subplot(width, height, i + 1)
        plt.imshow(faces[i])
        plt.axis('off')
    plt.show()


def plot_model_history(history):
    keys = history.history.keys()
    for key in keys: 
        # summarize history for accuracy
        plt.plot(history_test.history[key])
        plt.title(f'model {key}')
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

