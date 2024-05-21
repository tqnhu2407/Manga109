import manga109api

def load_all_images(parser, CLASSES):
    images = []
    for book in parser.books:
        annotation = parser.get_annotation(book=book)
        for i in range(len(annotation['page'])):
            temp = []
            image_path = parser.img_path(book=book, index=i)
            temp.append(image_path)
            temp.append(annotation["page"][i])
            images.append(temp)
    return images

def compute_mean_std(image_paths):
    n_images = len(image_paths)
    mean = np.zeros(3)
    std = np.zeros(3)

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0  # Convert to float and scale to [0, 1]
        mean += np.mean(image, axis=(0, 1))
        std += np.std(image, axis=(0, 1))

    mean /= n_images
    std /= n_images

    return mean, std