from train import run
from time import sleep


def main():

    conf_ths = [0.00]
    iou_ths = [0.70, 0.60, 0.50]

    for conf_th in conf_ths:
        for iou_th in iou_ths:
            run(imgsz=640, epochs=100, data='custom.yaml', weights='yolov5s.pt', cfg='models/yolov5s.yaml', patience=0, conf_th=conf_th, iou_th=iou_th)


if __name__ == "__main__":
    main()