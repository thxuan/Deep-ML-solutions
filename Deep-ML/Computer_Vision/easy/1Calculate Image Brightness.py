def calculate_brightness(img):
    if not img:
        return -1
    avg_brightness = 0
    n=0
    for imgs in img:
        if len(imgs) != len(img):
            return -1
        for i in range(len(imgs)):
            if imgs[i]<0 or imgs[i] > 255:
                return -1
            avg_brightness += imgs[i]
            n+=1            

    return(round(avg_brightness/n,2)) 

img = [
    [100, 200],
    [50, 150]
]
print(calculate_brightness(img))