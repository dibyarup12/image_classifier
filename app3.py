from flask import Flask, request
from flask_cors import CORS
import PIL
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models


def load_classifier(check_path):
    checkpoint = torch.load(check_path)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 400)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(400, 3)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.load_state_dict(checkpoint,strict=False)
    
    return model

SAVE_PATH = 'classifier.pth'

model=load_classifier('classifier.pth')


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


def process_image(image):
    img= PIL.Image.open(image)
    return test_transforms(img)

def predict(img_path,model):
    model.eval()
    img_pros=process_image(img_path)
    img_pros=img_pros.view(1,3,224,224)
    with torch.no_grad():
        output=model(img_pros)
        return output

app= Flask(__name__)
cors= CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/',methods=['GET'])
def home():
    return 'API is running'

@app.route('/pred',methods=['POST'])
def pred():
    im= request.files['img']
    log_ps=predict(im,model)
    cls_score=int(torch.argmax(torch.exp(log_ps)))
    if cls_score == 0:
        return('airplane')
    elif cls_score == 1:
        return('car')
    else:
        return('ship')
    
if __name__ == '__main__':
    app.run(port=8000, debug=True)