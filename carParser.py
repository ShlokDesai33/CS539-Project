
import xml.etree.ElementTree as ET
import os
from PIL import Image
import numpy as np
import matplotlib.patches as patches

class CarData:
    
    def __init__(self, path):
        self.path = path
        
    # Asynchronous loading
    def load_image(self):
        return np.asarray(Image.open(self.path))
    
    # Asynchronous loading
    def load_bbox(self):
        xml_file_path = self.path.replace('.jpeg', ".xml")
        return parseLicensePlateXML(xml_file_path)
    
    '''
    def __init__(self, path, image=None, bbox=None):
        self.path = path
        if image is None:
            imagePath = path
            self.image = np.asarray(Image.open(imagePath))
        else:
            if type(image)!="<class 'numpy.ndarray'>":
                self.image = np.asarray(image)
            else:
                self.image = image
        if bbox is None:  
            xmlFilePath = path.replace('.jpeg', ".xml")
            self.bbox = parseLicensePlateXML(xmlFilePath)
        else:
            self.bbox = bbox
    '''
            
    
class Rectangle:
    
    def __init__(self, minX, minY, maxX, maxY):
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
        
    def width(self):
        return self.maxX - self.minX
    
    def height(self):
        return self.maxY - self.minY
    
    def center(self):
        return [(self.minX + self.maxX)//2, (self.minY + self.maxY)//2]
    
    def numpy(self):
        return np.array([self.minX, self.minY, self.maxX, self.maxY])
    
    def fromNumpy(array):
        return Rectangle(array[0], array[1], array[2], array[3])
    


def parseLicensePlateXML(file):
    
    try:
        root = ET.parse(file).getroot()
    except FileNotFoundError as error:
        return Rectangle(0, 0, 0, 0)
    bndBox = root.find("object").find("bndbox")
    xMin = int(bndBox.find("xmin").text)
    yMin = int(bndBox.find("ymin").text)
    xMax = int(bndBox.find("xmax").text)
    yMax = int(bndBox.find("ymax").text)

    w = xMax-xMin
    h = yMax-yMin
    
    
    return Rectangle(xMin, yMin, xMax, yMax)

# Util function to convert bbox into Rectangle
def rectangle_patch(bbox, c='r'):
    return patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=c, facecolor='none')