
import cv2
import glob
import numpy as np
import os


class ImageProcessor:

    def __init__(self, file=None):
        
        self.filenm= os.path.basename(file)
        self.img=cv2.imread(file)

    def get_size(self):
        """
        Method to get the size of image               
        
        """
        self.h, self.w, self.c = self.img.shape

    
    def resize_images(self,target_height=None):
        """
        Method to resize images to target height

        Args:
            target_height: This is based on the desired height
            new_width:  This is width calculated maintaining the aspect ratio          

        Returns:
            resized image file returned to hardcoded pathway

        """

        self.get_size()        
        new_width = int((target_height * self.h)/self.w)
        #print(new_width,target_height)
        desiredsize = (new_width,target_height)
        output = cv2.resize(self.img,desiredsize)
        cv2.imwrite(f'processed_images/{self.filenm}',output)    

  

def load_and_resize_images(pth=None):
    """
    Method to load the downloaded images and work out the min height then resize the images
    Arg:

        height: list to store all the height of images 
        minheight: the smallest height of the scanned images

    Return:
            Results in resized images          
    
    """


    height = [] 
    for file in pth:
        #print(file)
        ip = ImageProcessor(file)
        ip.get_size()
        #print(ip.h)
        height.append(ip.h)
    minheight = min(height)

    for file in pth:
        ip = ImageProcessor(file)
        ip.resize_images(minheight)





if __name__ == "__main__":

    pth = [ i for i in glob.glob("images/*/*png")]
    load_and_resize_images(pth)
