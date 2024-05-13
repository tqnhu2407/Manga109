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
