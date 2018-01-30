import os
import cv2

IMAGE_DIR = './output'
OUTPUT_PATH = './output/output.avi'
if __name__ == '__main__':
  name_list = os.listdir(IMAGE_DIR)
  image_list = [name_list[i] for i in range(len(name_list)) if '.jpg' in name_list[i]]
  image_list.sort()

  test_path = os.path.join(IMAGE_DIR, image_list[0])
  test_frame = cv2.imread(test_path)
  height, width, channel = test_frame.shape

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (width, height), True)

  for image in image_list:
    frame_path = os.path.join(IMAGE_DIR, image)
    frame = cv2.imread(frame_path)
    print(image)
    out.write(frame)

  out.release()
    
