from ultralytics import YOLO
model = YOLO('best.pt')
results = model(source = 'guns.jpg', show=True, conf = 0.4, save = True)

print('code run successfull!')

