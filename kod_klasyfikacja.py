import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import copy
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transformacja danych
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train_data = ImageFolder(r"C:\Users\jakub\OneDrive\Pulpit\archive\Training", transform=transform)
test_data = ImageFolder(r"C:\Users\jakub\OneDrive\Pulpit\archive\Testing", transform=transform)


# Podział zbioru treningowego na train/validation
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)




# Model CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) #liczba 0 wokół) #lupa
        self.pool = nn.MaxPool2d(2, 2) #wybieranie maks wartosci
        self.flatten = nn.Flatten() #spłaszczenie do rzędu liczb
        self.dropout = nn.Dropout(p=0.2) #bedzie zerowal 20% elementów
        self.fc1 = nn.Linear(128 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 4)

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.flatten(x)
        features = self.fc1(x)
        return features

    def data_processing(self, x):
        features = self.get_features(x)
        x = F.relu(features)
        x = self.fc2(x)
        return x



net = CNNModel().to(device)
kryt = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
best_val_loss = np.inf #nieskonczonosc, żeby kazda liczba byla lepsza od tego

# Early stopping params (patience and counter):
patience = 3 #tyle razy możemy wytrzymać pogorszenie loss
counter = 0 # ile nie było poprawy 
best_model_weights = copy.deepcopy(net.state_dict())

for epoch in range(15):
    net.train()

    train_loss = 0 #ile razy sie pomylil podczas treningu
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs) #wyniki z sieci
        loss=kryt(outputs, labels) #liczymy strate
        loss.backward() #liczymy gradient, jak poprawic wagi
        optimizer.step() #uzupełnia wagi
        train_loss += loss.item()

    net.eval()
    val_loss = 0 #ile razy sie pomylił podczas walidacji
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = kryt(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch}: training loss={train_loss}, validation loss={val_loss}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1 #powiekszamy counter, jak dobije do maks lossu to early stop 
        print(f"Validation loss did not improve {counter} times")

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

net.load_state_dict(torch.load('best_model.pth'))




X_train, y_train = [], []
net.eval()
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        features = net.get_features(inputs)
        X_train.append(features.detach().cpu().numpy())
        y_train.append(labels.detach().cpu().numpy())

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

X_test, y_test, p_test = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        features = net.get_features(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        X_test.append(features.cpu().numpy())
        p_test.append(probs)
        y_test.append(labels.cpu().numpy())

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)
p_test = np.concatenate(p_test)




# ACC AUC dla CNN
ACC_cnn = np.mean(y_test == np.argmax(p_test, axis=1)) #accuracy
AUC_cnn = roc_auc_score(y_test, p_test, multi_class="ovr") #Area under curve

# Metryka zew klasyfikatora
y_pred_clf = clf.predict(X_test)
y_pred_clf_proba = clf.predict_proba(X_test)
print(f"y_pred_clf = {y_pred_clf}")
print(f"y_pred_clf_proba = {y_pred_clf_proba}")
ACC_clf = np.mean(y_test == y_pred_clf)
AUC_clf = roc_auc_score(y_test, y_pred_clf_proba, multi_class="ovr") #Area Under Curve czyli jak dobrze klasyfikuje do klas

print(f"CNN: ACC={ACC_cnn:.4f}, AUC={AUC_cnn:.4f}")
print(f"clf: ACC={ACC_clf:.4f}, AUC={AUC_clf:.4f}")




def plot_roc_multiclass(y_true, y_score, n_classes, title="ROC Curve"):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # linia losowego zgadywania
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

n_classes = 4
plot_roc_multiclass(y_test, p_test, n_classes, title="ROC Curve for CNN")

# Dla klasyfikatora logistycznego
plot_roc_multiclass(y_test, y_pred_clf_proba, n_classes, title="ROC Curve for Logistic Regression")

metrics_names = ['Accuracy', 'AUC']
cnn_metrics = [ACC_cnn, AUC_cnn]
clf_metrics = [ACC_clf, AUC_clf]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, cnn_metrics, width, label='CNN')
rects2 = ax.bar(x + width/2, clf_metrics, width, label='Logistic Regression')

ax.set_ylabel('Wartość')
ax.set_title('Porównanie metryk modeli')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.1)
ax.legend()
ax.bar_label(rects1, fmt='%.3f')
ax.bar_label(rects2, fmt='%.3f')

plt.show()

#Macierze pomyłek
def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

y_pred_cnn = np.argmax(p_test, axis=1)

plot_confusion(y_test, y_pred_cnn, title="Confusion Matrix CNN")
plot_confusion(y_test, y_pred_clf, title="Confusion Matrix Logistic Regression")