### 第0步：环境准备
##检查是否有nvida显卡
#nvidia-smi

##创建激活conda环境
#conda create -n pytorch_gpu python=3.10 -y
#conda activate pytorch_gpu

##装好几个库
#pytorch：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
#torchvision
#tqdm 
#matplotlib

###第1步：数据下载预处理
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 使用 GPU 或 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理：转张量 + 归一化到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 数据路径
data_path = './data/'

# 下载训练集和测试集
try:
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
except Exception as e:
    print("下载失败，可能是网络问题，请手动下载 MNIST 数据集到 ./data/")
    raise e

# 每个 batch 的大小
BATCH_SIZE = 256

# 构建 DataLoader（训练集打乱，测试集不打乱）
trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

### 第2步：定义一个训练过程
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            #The size of the picture is 28x28
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self,input):
        output = self.model(input)
        return output
net = Net()
#将模型转换到device中，并将其结构显示出来
print(net.to(device))  

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

EPOCHS = 10
###第3步：存储训练过程：把数据传输进设备，梯度清零，前向传播，计算损失以及正确率，反向传播，优化梯度
#创建一个字典存储测试的损失和正确率
history = {'Test Loss':[],'Test Accuracy':[]}
#迭代
for epoch in range(1,EPOCHS + 1):
    processBar = tqdm(trainDataLoader,unit = 'step')
    #打开网络的训练模式
    net.train(True)
    for step,(trainImgs,labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs,labels)
        predictions = torch.argmax(outputs, dim = 1)
        accuracy = torch.sum(predictions == labels)/labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item()))
        #测试集
        if step == len(processBar)-1:
            correct,totalLoss = 0,0
            net.train(False)
            for testImgs,labels in testDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs,labels)
                predictions = torch.argmax(outputs,dim = 1)
                
                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct/(BATCH_SIZE * len(testDataLoader))
            testLoss = totalLoss/len(testDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()

plt.plot(history['Test Loss'], label='Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss over Epochs')
plt.savefig('loss_curve.png')  # ✅ 保存图像
plt.close()

# 对测试准确率进行可视化
plt.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.savefig('accuracy_curve.png')  # ✅ 保存图像
plt.close()







