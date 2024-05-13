import manga109api

def main():
    with open('out.txt', 'w') as f:
        # (1) Book titles 
        f.write('(1) Book titles\n\n')
        f.write(str(p.books))
        f.write('\n\n')

        # (2) Path to an image (page)
        f.write('(2) Path to an image (page)\n\n')
        f.write(p.img_path(book="ARMS", index=3))
        f.write('\n\n')

        # (3) The main annotation data
        f.write('(3) The main annotation data\n\n')
        annotation = p.get_annotation(book="ARMS")

        f.write('(3-a) Title\n\n')
        f.write(annotation["title"])
        f.write('\n\n')

        f.write('(3-b) Character\n\n')
        f.write(str(annotation["character"]))
        f.write('\n\n')

        f.write('(3-c) Page\n\n')
        f.write(str(annotation["page"][3]))
        f.write('\n\n')

        # (4) Preserve the raw tag ordering in the output annotation data
        f.write('(4) Preserve the raw tag ordering in the output annotation data\n\n')
        annotation_ordered = p.get_annotation(book="ARMS", separate_by_tag=False)
        f.write(str(annotation_ordered['page'][3]))

if __name__ == "__main__":
    manga109_root_dir = "./Manga109_released_2023_12_07"
    p = manga109api.Parser(root_dir=manga109_root_dir)
    annotation = p.get_annotation(book="ARMS")
    print('annotation:', annotation['page'][0])
