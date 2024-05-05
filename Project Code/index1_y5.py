from ultralytics import YOLO
model = YOLO('best_y5.pt')
# results = model(source = 'one.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'two.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'three.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'four.jpg', show=True, conf = 0.4, save = True)
results = model(source = 'guns5.jpeg', show=True, conf = 0.4, save = True)


print('code run successful!')