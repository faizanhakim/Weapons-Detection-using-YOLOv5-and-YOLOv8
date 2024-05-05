from ultralytics import YOLO
model = YOLO('best_y8.pt')
# results = model(source = 'one.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'two.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'three.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'four.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'guns.jpg', show=True, conf = 0.4, save = True)
# results = model(source = 'guns(2).jpeg', show=True, conf = 0.4, save = True)
# results = model(source = 'guns(3).jpeg', show=True, conf = 0.4, save = True)
# results = model(source = 'guns1.jpeg', show=True, conf = 0.4, save = True)
results = model(source = 'guns5.jpeg', show=True, conf = 0.4, save = True)
# results = model(source = 'linggis.jpeg', show=True, conf = 0.4, save = True)
# results = model(source = 'celurit.jpeg', show=True, conf = 0.4, save = True)





print('code run successful!')
