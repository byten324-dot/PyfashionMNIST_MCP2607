import streamlit as st
import torch
from torch import nn
import torchvision.transforms as transforms
import os
import json

from PIL import Image
import PIL.ImageOps as ops


# ==========================================
# 1. CNN 모델 정의
# ==========================================
class MyCNNModel(nn.Module):

    def __init__(self):
        """모델에 사용되는 레이어들 정의"""
        super().__init__()

        # 입력: 1채널(흑백) 28×28 이미지
        # conv1: 1채널 → 32채널
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # conv2: 32채널 → 64채널
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Max Pooling
        self.pooling = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 완전연결 은닉층
        self.fc1 = nn.Linear(
            in_features=64 * 7 * 7,
            out_features=256
        )

        # 완전연결 출력층
        # FashionMNIST는 총 10개의 클래스
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=10
        )

        # Dropout
        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p=0.5)


    def forward(self, data):
        """모델 입력 data의 순전파 정의"""

        # --------------------------
        # Feature Extraction
        # --------------------------

        # 첫 번째 합성곱
        data = self.conv1(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        # 두 번째 합성곱
        data = self.conv2(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        # --------------------------
        # Classification
        # --------------------------

        # 1차원으로 펼치기
        data = data.view(-1, 7 * 7 * 64)

        # Fully Connected Layer
        data = self.fc1(data)
        data = torch.relu(data)
        data = self.dropout50(data)

        # 최종 10개 클래스에 대한 logit
        logits = self.fc2(data)

        return logits


# ==========================================
# 2. FashionMNIST 클래스
# ==========================================
classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


# ==========================================
# 3. 이미지 예측 함수
# ==========================================
def predict(file_path, model, transform):

    # 이미지 불러오기
    img = Image.open(file_path)

    # 흑백 이미지로 변환
    mono8img = img.convert('L')

    # 색상 반전
    invImg = ops.invert(mono8img)

    # 이미지 전처리
    transformed_img = transform(invImg)

    # 배치 차원 추가
    img_tensor = transformed_img.unsqueeze(0)

    # GPU 사용 가능하면 GPU 사용
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    img_tensor = img_tensor.to(device)

    # 평가 모드
    model.eval()

    # 예측 수행
    with torch.no_grad():

        logits = model(img_tensor)

        # 확률 계산
        probs = torch.softmax(logits, dim=1)

        # 가장 높은 확률을 가진 클래스
        predicted_class = torch.argmax(
            probs,
            dim=1
        ).item()

    return predicted_class, classes[predicted_class]


# ==========================================
# 4. Device 설정
# ==========================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 5. 학습된 모델 불러오기
# ==========================================
model_path = 'CNN_FashionMNIST2.pth'

model = MyCNNModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE
    )
)

# 평가 모드로 변경
model.eval()


# ==========================================
# 6. 이미지 전처리 설정 불러오기
# ==========================================
transform_config_path = 'CNN_FashionMNIST2_transform_config.json'

with open(transform_config_path, 'r') as f:
    transform_config = json.load(f)


# Transform 복원
transform = transforms.Compose([

    transforms.Resize(
        transform_config['resize']
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=transform_config['mean'],
        std=transform_config['std']
    )
])


# ==========================================
# 7. 업로드한 파일 저장 함수
# ==========================================
def save_uploaded_file(directory, file):

    # images 폴더가 없으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 저장 경로
    file_path = os.path.join(
        directory,
        file.name
    )

    # 파일 저장
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())

    return file_path


# ==========================================
# 8. Streamlit 화면
# ==========================================
st.title('FashionMNIST 이미지 분류')

st.write(
    '의류 이미지를 업로드하면 CNN 모델이 종류를 예측합니다.'
)

img_file = st.file_uploader(
    '이미지를 업로드 하세요',
    type=['png', 'jpg', 'jpeg']
)


# ==========================================
# 9. 이미지가 업로드된 경우
# ==========================================
if img_file is not None:

    # 업로드한 파일 저장
    file_path = save_uploaded_file(
        'images',
        img_file
    )

    # 업로드 이미지 화면에 표시
    st.image(
        file_path,
        caption='업로드한 이미지'
    )

    # CNN 모델로 예측
    pred_index, pred_class = predict(
        file_path,
        model,
        transform
    )

    # 결과 출력
    st.subheader('예측 결과')

    st.success(
        f'예측된 의류: {pred_class}'
    )
```
