import carParser
from scipy.ndimage import rotate
import numpy as np
import math
import tensorflow as tf
import cv2


def rotateCar(carData, degrees):
    newImage = carData.image
    pivot = carData.bbox.center()
    # rotate image
    newImage = rotate_image(newImage, angle=degrees, pivot=pivot)
    # rotated bounding box axis aligned
    bbox = rotate_bounding_box(carData.bbox, degrees, pivot)
    
    return carParser.CarData(carData.path, newImage, bbox)


def rotate_bounding_box(bbox, angle, pivot):
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)

    # Calculate the sine and cosine of the angle
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # Calculate the coordinates of the rotated corners
    corners = [(bbox.minX, bbox.minY), (bbox.maxX, bbox.minY),
               (bbox.maxX, bbox.maxY), (bbox.minX, bbox.maxY)]

    rotated_corners = []
    for x, y in corners:
        # Translate the corners to be relative to the pivot point
        translated_x = x - pivot[0]
        translated_y = y - pivot[1]

        # Perform the rotation
        new_x = translated_x * cos_angle - translated_y * sin_angle
        new_y = translated_x * sin_angle + translated_y * cos_angle

        # Translate the corners back to their original position
        rotated_x = new_x + pivot[0]
        rotated_y = new_y + pivot[1]

        rotated_corners.append((rotated_x, rotated_y))

    # Find the new bounding box coordinates from the rotated corners
    new_minX = min(x for x, _ in rotated_corners)
    new_maxX = max(x for x, _ in rotated_corners)
    new_minY = min(y for _, y in rotated_corners)
    new_maxY = max(y for _, y in rotated_corners)

    # Create a new Rectangle object with the rotated bounding box coordinates
    rotated_bbox = carParser.Rectangle(new_minX, new_minY, new_maxX, new_maxY)

    return rotated_bbox


def rotate_image(image, angle, pivot=None):
    if pivot is None:
        pivot = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_preprocess_function(target_height, target_width):
  def preprocess_image(image, label):
      # Resize the image to the required input shape
      resized_image = tf.image.resize_with_pad(image=image, target_height=target_height, target_width=target_width)
      
      # Normalize the pixel values to the range [0, 1]
      resized_image = resized_image / 255.0
      
      
      # Check if transform label calculations doesn't accidentally decrease the size of the bounding box or move it slightly
      # transform the label
      resize_scale_width = target_width / image.shape[1]
      resize_scale_height = target_height / image.shape[0]

      # Determine the resize scale to use (minimum or maximum)
      resize_scale = min(resize_scale_width, resize_scale_height)
      
      # Calculate the padding offsets
      pad_offset_height = (target_height - image.shape[0] * resize_scale) // 2
      pad_offset_width = (target_width - image.shape[1] * resize_scale) // 2

      xmin_resized = (label[0] * resize_scale) + pad_offset_width
      ymin_resized = (label[1] * resize_scale) + pad_offset_height
      xmax_resized = (label[2] * resize_scale) + pad_offset_width
      ymax_resized = (label[3] * resize_scale) + pad_offset_height
      newLabel = np.array([xmin_resized, ymin_resized, xmax_resized, ymax_resized])
      return resized_image, newLabel

  return preprocess_image
