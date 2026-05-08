import torch #파이토치
import torchvision #데이터셋
import torch.nn as nn #신경망 부품
from torchvision import datasets, transforms #데이터셋과 전처리 불러오기
import torch.optim as optim #옵티마이저
import matplotlib.pyplot as plt #그래프 도구
from torchmetrics import MetricCollection, Accuracy, F1Score #정확도 측정 도구
from tqdm import tqdm #학습 로딩바
from torch.utils.data import DataLoader

device = "mps" if torch.backends.mps.is_available() else "cpu" #mps(맥북 그래픽카드 가속 이름)이 발견되면 mps 아니면 cpu
print(f"현재 사용중인 기기:{device}")
EPOCH = 70 #학습 반복 횟수
BATCH_SIZE = 64 #모델에게 이미지를 한번에 몇장씩 묶어서 줄건지 보통 2의 제곱 사용
LEARNING_RATE = 0.001
#모델이 학습을 얼마나 빨리 할지 이 값이 높으면 학습이 빠르지만 제대로 되지 않음 이 값이 낮으면 학습은 느리거나 안되지만 정확도가 높아짐
#쉽게 말하면 보폭, 예를 들자면 사람이 10미터 앞 목적지에 최대한 가까이 가는것(정확도를 높이는것)이 목표라 한다면 보폭(러닝레이트)가 높을수록 빠르게 목적지로 갈수있지만
#10미터를 넘어가버리기 쉬움 하지만 보폭이 작다면 목적지까지 가는 시간이 느린 대신 더 정확히 10미터에 멈출 수 있음
P = 0.3
#드롭아웃(모델이 학습 데이터를 외워버리는걸 막기 위해 신경망 몇개를 강제로 꺼버리는데 그 확률 저건 30퍼센트라는 뜻

transform = transforms.Compose([ #compose는 데이터 전처리를 한 모듈로 묶어주는 역할
    transforms.Resize((28, 28)),  #데이터 사이즈를 28 x 28로 만들어줌
    transforms.ToTensor(), #데이터를 텐서로 변환해줌
    transforms.Normalize((0.1307,), (0.3081,)) #데이터 정규화(그리 중요한건 아님)
])

def visualize(model=None, device=None, DL=None):
#ai가 데이터를 학습할때 어떤 특징을 학습하는지 보게 해주는 함순데 굳이 할 필요 없음 나중에 알려줄게
    model = model.to(device)
    sample, labels = next(iter(DL))  # b,c,h,w
    sample = sample[0].unsqueeze(0).to(device)  # b,c,h,w
    labele = labels[0].item()

    output = model.linear1(sample)  # b,c,h,w
    output = output.squeeze(0).cpu().numpy()  # c,h,w
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f"Layer 1 Feature Maps (Input Label: {labele})", fontsize=16, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < output.shape[0]:
            ax.imshow(output[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Filter {i + 1}")
    plt.tight_layout()
    plt.show()


train_mnist_dataset = datasets.MNIST('dataset', train=True, download=True, transform=transform) #데이터를 전처리 시켜 묶어주는 역할
test_mnist_dataset = datasets.MNIST('dataset', train=False, download=True, transform=transform)
#train과 download는 모듈에 내장되어있는 데이터를 쓰는거라 저거고 내 데이터를 가지고 학습을 시키고 싶으면
#torch.utils.data.DataSet(root=' ', tramsform=)이렇게

train_DL = DataLoader(train_mnist_dataset, batch_size=512, shuffle=True) #데이터셋을 배치사이즈에 맞게 세팅해주는 함수
test_DL = DataLoader(test_mnist_dataset, batch_size=512, shuffle=False)
#학습용은 순서를 외워버릴수도 있기 때문에 shuffle(데이터 순서 섞기)를
#활성화 해줘야하고 테스트 데이터셋은 항상 일정한 기준을 가지고 모델에 들어가야 정확도 차이를 비교해볼수 있으므로 shuffle은 꺼야함


class MyModel(nn.Module): #nn.Module을 상속
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Sequential( #senquential은 여러 신경망 부품을 묶어주는 역헐
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

    def forward(self, x): #여기서 만들어둔 신경망 모듈을 조립
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


model = MyModel().to(device) #모델을 gpu로 옮김(gpu 가속을 위해)
criterion = nn.CrossEntropyLoss() #손실함수
optimizer = optim.AdamW(model.parameters(), lr=0.01) #옵티마이저
model.train() #학습모드(드롭아웃이 켜짐)

test_loss_history = [] #그래프 그리기 위해 값을 기록해둘 빈 리스트
for epoch in range(EPOCH): #반복횟수동안 학습 반복
    model.train()
    process_bar = tqdm(train_DL, desc=f'{epoch + 1}/{EPOCH}', colour='green') #학습 진행상태 체크 표시 함수
    total_train_loss = 0
    total_test_loss = 0
    avg_train_loss = 0
    avg_test_loss = 0
    for x, y in process_bar: #데이터 로더에서 데이터와 정답값 가져오기
        x, y = x.to(device), y.to(device) #데이터와 정답값도 gpu로 보내줘야지 학습 가능(모델이 gpu에 위치해있기 때문)
        output = model(x) #모델 결과값
        loss = criterion(output, y) #결과값과 정답값 비교후 오차 loss 변수에 저장
        optimizer.zero_grad() #가중치 기록 초기화(초기화 안하면 저번 에포치 기록까지 반영되기 때문)
        loss.backward() #어느 부분에서 오차가 크게 증가했는지(신경망 부품별로 누가 책임이 큰지, 어떤거의 가중치를 바꿔야 loss값이 크게 줄어드는지 파악)
        optimizer.step() #가중치 업데이트
        tqdm.set_postfix(process_bar, loss=loss.item()) #진행바 업데이트(신경안써도됨)
        total_train_loss += loss.item() #토탈 loss값에 현재 loss값 추가

    with torch.no_grad(): #테스트 코드(이때는 가중치 업데이트를 안해도 되기 떄문에 가중치 기록 다 끄는 코드)
        model.eval() #평가모드(드롭아웃 해제)
        test_process_bar = tqdm(test_DL, desc=f'{epoch + 1}', colour='red')
        for x, y in test_process_bar: #평가 데이터로더에서 데이터와 정답값 가져오기
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_test_loss += loss.item()
            tqdm.set_postfix(test_process_bar, loss=loss.item())

        visualize(model=model, device=device, DL=test_DL)

    avg_train_loss = total_train_loss / len(train_DL)
    avg_test_loss = total_test_loss / len(test_DL)
    tqdm.write(f"\n{avg_train_loss:.3f}")
    tqdm.write(f"{avg_test_loss:.3f}")



