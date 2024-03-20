import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models.resnet import ResNet50_Weights
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

train_data_path = './EE6222 train and validate 2024/train.txt'
validate_data_path = './EE6222 train and validate 2024/validate.txt'
path_train = './EE6222 train and validate 2024/train/'
path_validate = './EE6222 train and validate 2024/validate/'
video_paths_train = []
video_paths_validate = []
path_o_train = './original_train/'
path_v_validate = './original_validate/'
save_path_train = './gamma_train/'
save_path_validate = './gamma_validate/'
gamma1 = 1.2
path_label_train = {}
train_label = []
validate_label = []
path_label_validate = {}
with open(train_data_path, 'r') as file:
    for line in file:
        components = line.strip().split('\t')
        video_id, class_label, video_path = components
        video_paths_train.append(path_train + video_path)
        path_label_train[path_train + video_path] = int(class_label)

with open(validate_data_path, 'r') as file:
    for line in file:
        components = line.strip().split('\t')
        video_id, class_label, video_path = components
        video_paths_validate.append(path_validate + video_path)
        path_label_validate[path_validate + video_path] = int(class_label)

frames_train = []
all_features_train = []
frames_validate = []
all_features_validate = []


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def uniform_sample(video_path1, list1, sampling_rate):
    cap = cv2.VideoCapture(video_path1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for j in range(0, total_frames, sampling_rate):
        # Set the current frame position of the video file
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ret, frame = cap.read()
        if ret:
            gamma_corrected = adjust_gamma(frame, gamma=gamma1)
            rgb_frame = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
            list1.append(rgb_frame)
    cap.release()


def random_sampling(video_path1, list1, num_samples):
    cap = cv2.VideoCapture(video_path1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.random.choice(total_frames, size=num_samples, replace=False)
    frame_indices.sort()
    for idx, frame_index in enumerate(frame_indices):
        # Set the current frame position of the video file to the sample index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            gamma_corrected = adjust_gamma(frame, gamma=gamma1)
            rgb_frame = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
            list1.append(rgb_frame)
    cap.release()


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return hn[-1]


def extract_features(frame_s):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        list1 = []
        for frame1 in frame_s:
            frame = Image.fromarray(frame1)
            input_tensor = preprocess(frame).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.to('cuda')
                model.to('cuda')
            output = model(input_tensor)
            output = output.cpu().flatten()
            list1.append(output)

        spatial_features_tensor = torch.stack(list1)
        input_size = 2048  # Size of input features
        hidden_size = 512  # LSTM hidden layer size
        num_layers = 3
        lstm_feature_extractor = LSTMFeatureExtractor(input_size, hidden_size, num_layers)
        lstm_features = lstm_feature_extractor(spatial_features_tensor)
        return lstm_features


def main():
    for i in video_paths_train:
        list1 = []
        # uniform_sample(i, list1, 5)
        random_sampling(i, list1, 6)
        frames_train.append(list1)
        train_label.append(path_label_train[i])

    for i in frames_train:
        all_features_train.append(extract_features(i))

    x_train = np.array(all_features_train)
    y_train = np.array(train_label)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_pca = pca.fit_transform(features_scaled)
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_pca, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    X_selected = X_pca[:, indices[:30]]
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'kernel': ['linear']
    }
    clf = SVC()
    grid_search = GridSearchCV(clf, param_grid, refit=True, verbose=2, cv=10)
    grid_search.fit(X_selected, y_train)
    print("最佳参数组合: ", grid_search.best_params_)
    print("最佳模型分数: ", grid_search.best_score_)
    for i in video_paths_validate:
        list1 = []
        # uniform_sample(i, list1, 5)
        random_sampling(i, list1, 6)
        frames_validate.append(list1)
        validate_label.append(path_label_validate[i])

    for i in frames_validate:
        all_features_validate.append(extract_features(i))

    x_validate = np.array(all_features_validate)
    y_validate = np.array(validate_label)
    scaler = StandardScaler()
    features_scale = scaler.fit_transform(x_validate)
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_pca = pca.fit_transform(features_scale)
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_pca, y_validate)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    X_selected = X_pca[:, indices[:30]]
    predictions = grid_search.predict(X_selected)
    print(classification_report(y_validate, predictions, zero_division=1))


if __name__ == "__main__":
    main()
