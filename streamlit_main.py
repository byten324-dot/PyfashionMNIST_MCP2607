import streamlit as st
import torch
from torch import nn
import torchvision.transforms as transforms
import os, json



# model 정의
class MyCNNModel(nn.Module):

  def __init__(self):
    """모델에 사용되는 레이어들 정의"""
    super().__init__()
    # 입력: 1채널(흑백) 28×28 이미지.
    # conv1: (1→32채널, 3×3 커널), 출력 28×28.
    self.conv1 = nn.Conv2d(in_channels=1,  # 입력 채널 개수
                           out_channels=32, # 출력 채널 개수 (필터의 개수)
                           kernel_size=3,
                           padding=1)

    # conv2 (32->64채널, 3x3커널), 출력 14x14
    self.conv2 = nn.Conv2d(in_channels=32,
                           out_channels=64,
                           kernel_size=3, padding=1)

    # pooling: 크기 절반
    self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    # 완전연결 (은닉층)
    self.fc1 = nn.Linear(in_features=64 * 7 * 7,
              out_features=256)   # 출력, 뉴런의 개수.

    # 완전연결 (출력층)
    self.fc2 = nn.Linear(in_features=256, out_features=10)

    # Dropout
    self.dropout25 = nn.Dropout(p=0.25)
    self.dropout50 = nn.Dropout(p=0.5)


  def forward(self, data):
    """모델 입력 data 의 순전파 정의"""
    # [Feature Extraction]
    # 합성곱 → ReLU → 풀링 → 드롭아웃 순서로 반복
    data = self.conv1(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.dropout25(data)

    data = self.conv2(data)
    data = torch.relu(data)
    data = self.pooling(data)
    data = self.dropout25(data)

    # [Classification]
    # 마지막에 fully connected layer 2개를 거쳐 10 클래스에 대한 logit 출력.
    data = data.view(-1, 7 * 7 * 64)   # 1차원으로 펼치기

    data = self.fc1(data)
    data = torch.relu(data)
    data = self.dropout50(data)

    logits = self.fc2(data)

    return logits

# 함수 만들
from PIL import Image
import PIL.ImageOps as ops

classes = ['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']

def predict(file_path, model, transform):
    img = Image.open(file_path)
    mono8img = img.convert('L')
    invImg = ops.invert(mono8img)

    transformed_img = transform(invImg)
    img_tensor = transformed_img.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 모델도,
    img_tensor = img_tensor.to(device)  # 입력 텐서도.

    # 예측 수행
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return predicted_class, classes[predicted_class]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
model_path = 'CNN_FashionMNIST2.pth'
model = MyCNNModel().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))

# 전처리 불러오기
transform_config_path = 'CNN_FashionMNIST2_transform_config.json'
transform_config = json.load(open(transform_config_path, 'r'))

# transform 복원
transform = transforms.Compose([
    transforms.Resize(transform_config['resize']),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_config['mean'], std=transform_config['std'])
])

# 파일 업로드 함수
def save_uploaded_file(directory, file):
  if not os.path.exists(directory):
    os.makedirs(directory)

  with open(os.path.join(directory, file.name), 'wb') as f:
    f.write(file.getbuffer())

  return st.success("파일 업로드 성공")

st.title('fashionMNIST')

img_file = st.file_uploader('이미지를 업로드 하세요', type=['png', 'jpg', 'jpeg'])


if img_file:
  same_uploaded_file('images', img_file)
  st.image(f'images/{img_file.name}')

  _, pred_class = predict(os.path.join('images', img_file.name), model, transform)
  st.subheader(pred_class)
