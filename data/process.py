from pathlib import Path
import sys
from PIL import Image

size = (64,64)

for i in range(1, len(sys.argv)):
    for p in Path(sys.argv[i]).iterdir():
        try:
            im = Image.open(str(p))
            im.thumbnail(size, Image.ANTIALIAS)
            background = Image.new('RGBA', size, (255,255,255,0))
            background.paste(im,(int((size[0] - im.size[0]) // 2),int((size[1] - im.size[1]) / 2)))
            background.save(str(p), "bmp")
        except IOError:
            print(f"Failed to process {infile}!")
