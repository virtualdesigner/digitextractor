from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS, cross_origin
import re
from PIL import Image
from io import BytesIO
import base64
import binascii
import torch
import torch.nn as nn
import math
import os

app = Flask(__name__)
CORS(app)

# Display your index page


@app.route("/")
def index():
    return render_template('index.html')

# A function to add two numbers


@app.route("/predict", methods=['GET', 'POST', 'HEAD'])
def predict():
    # print('woah!', a, b)
    image_b64 = request.get_json(force=True)['imgData']
    image_data = re.sub('^data:image/.+;base64,', '',
                        image_b64)
    image_PIL = Image.open(
        BytesIO(base64.b64decode(image_data))).convert('LA').resize((28, 28))
    # print(image_PIL)

    return jsonify({"prediction": str(predictLogic(image_PIL))})


def predictLogic(pil_img):
    model = Model()
    # print(list(model.parameters()))

    model.load_state_dict(torch.load(
        'model_parameters3.pth', map_location=torch.device('cpu')))

    # print(torch.sum(torch.tensor(list(pil_img.getdata()), dtype=torch.float64), 1).shape)
    test_image = torch.sum(torch.tensor(
        list(pil_img.getdata()), dtype=torch.float64), 1).view(1, 1, 28, 28)
    test_res = torch.tensor([3]).type(torch.long)
    test_pred, test_loss = model.forward(
        test_image.float(), test_res)
    return torch.nn.functional.softmax(test_pred).detach().cpu().argmax().item()


class Lin(torch.autograd.Function):
    @staticmethod
    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)
#         print(input.shape, weight.shape, bias.shape)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(self, next_grad):
        input, = self.saved_tensors
#         print('backward', input.shape)
        return input


class Linear(torch.nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out

        self.weight = torch.nn.Parameter(
            torch.randn(fan_out, fan_in) * math.sqrt(2/fan_in))
#         if bias == True:
        self.bias = torch.nn.Parameter(
            torch.randn(fan_out) * math.sqrt(2/fan_in))
#         else:
#             self.bias = False

    def forward(self, x):
        # See the autograd section for explanation of what happens here.
        #         return Lin.apply(input.view(input.shape[0], input.shape[1]*input.shape[2]*input.shape[3]), self.weight, self.bias)
        res = x.view(x.shape[0], self.weight.shape[1]) @ self.weight.t()
        if self.bias is not False:
            res += self.bias
#         print('wow!', res.shape)
        return res


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(3),  # 5 x 5 x 4
            Linear(2304, 1500, 1500),
            nn.ReLU(),
            Linear(1500, 500, 500),
            nn.ReLU(),
            Linear(500, 10, 10),
        ])
        # need to implement softmax? No, CrossEntropyLoss will take care of it.
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        for l in self.layers:
            x = l(x)
#         print(x)
        preds = x
        loss = self.loss(x, y)
#         print(x.shape)
        return preds, loss


if __name__ == "__main__":
    # app.run(debug=False, port=os.getenv('PORT', 5000))
    app.run(debug=True)
