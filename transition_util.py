import os


def get_center_coord_height_and_width(top_left_corner_x, top_left_corner_y, bot_right_corner_x, bot_right_corner_y, image_width, image_height):

    width = bot_right_corner_x - top_left_corner_x
    height = bot_right_corner_y - top_left_corner_y

    center_x = top_left_corner_x +  width / 2
    center_y = top_left_corner_y + height / 2


    width /= image_width
    height /= image_height

    center_x /= image_width
    center_y /= image_height

    return center_x, center_y, width, height


def get_yolo_format_from_csv(annotations_file_path, image_width, image_height, class_to_id):
    '''
    Make YOLO format from CSV data format

    In csv file each line represents:
    "47_superhikov_veliki_poduhvat_022_high.jpg",397.48494983277595,22.563741721854303,550.9565217391305,152.96771523178808,"Broj 1"
        image_name, top_left_corner_x, top_left_corner_y, bot_right_corner_x, bot_right_corner_y in px values
    
    YOLO format should have one txt file per image with data (.txt file name is the name of the image):
        class_id, center_x, center_y, width, height
        Where latter 4 coordinates are in ranges of [0, 1]
    '''

    output_file_name = 'yolo_output5'
    if not os.path.exists(output_file_name):
        os.mkdir(output_file_name)

    annotation_file = open(annotations_file_path, "r")
    lines = annotation_file.read().splitlines()

    last_image_name = ' '
    txt_file = None

    for line in lines[1:]: # Skipping first line which has column names
        image_info = line.split(',')
        image_name = (image_info[0].strip('\"'))[:-4] # removing quotes from string and removing '.png' ext from name
        top_left_corner_x = int(float(image_info[1]))
        top_left_corner_y = int(float(image_info[2]))
        bot_right_corner_x = int(float(image_info[3]))
        bot_right_corner_y = int(float(image_info[4]))
        label = image_info[5].strip('\"')

        center_x, center_y, width, height = get_center_coord_height_and_width(top_left_corner_x, top_left_corner_y, bot_right_corner_x, bot_right_corner_y, image_width, image_height)
        class_id = class_to_id[label]


        if last_image_name != image_name:  # New image so we make a file for it
            last_image_name = image_name

            image_txt_file_path = os.path.join(output_file_name, image_name + '.txt')
            txt_file = open(image_txt_file_path, 'w+')
        
        txt_file.write(f'{class_id} {center_x} {center_y} {width} {height} \n')


def main():
    classes = ['DrHouse', 'DrWilson', 'Foreman', 'Chase', 'Cameron', 'Cuddy']

    class_to_id = {class_name : class_index for class_index, class_name in enumerate(classes)}
    
    image_width = 1280
    image_height = 720
    annotations_file_path = "/media/workstation/Disk 1/drh/drh/annotations/annotations.csv"
    annotations_file_path = '/media/workstation/Disk 1/drh/drh/annotations/DrHouse5-export.csv'
    get_yolo_format_from_csv(annotations_file_path, image_width, image_height, class_to_id)


if __name__ == '__main__':

    main()