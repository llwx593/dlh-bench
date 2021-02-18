import pretrained_models as pm

model = pm.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
model.eval()