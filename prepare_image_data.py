#%%
import tabular_data
import os, glob
from PIL import Image

def resize_images(df):

    for property in range(0,4): #running for only first 4 properties
        id = df['ID'][property]
        image_dir = f"images/{id}/*.png"
        lst = glob.glob(image_dir)
        
        os.makedirs(f"tabular_data/processed_images/{id}", exist_ok=True)
        
        #looping over the images in the directory
        for pic in lst:  
            im = Image.open(pic)

            if im.mode=='RGB':
            
                im_width, im_height = im.size
                aspect_ratio = im_width/im_height
                new_height = 400
                new_width = int(aspect_ratio * new_height)
                im_resized = im.resize((new_height, new_height))

                outfile = f"tabular_data/processed_images/{id}/{pic[44:]}"
                im_resized.save(outfile)

if __name__ == "__main__":
    file = "clean_tabular_data.csv"
    df = tabular_data.read_csv_data(file) 

    os.makedirs('tabular_data/processed_images', exist_ok=True)

    resize_images(df) 