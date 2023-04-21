import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from pytorch.clothes_classify_train_model import NeuralNetwork

loadedModel = NeuralNetwork()
loadedModel.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

loadedModel.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loadedModel(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')