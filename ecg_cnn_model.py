import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def build_and_train_model(features, labels, num_classes=None, epochs=20, batch_size=32):
    """
    构建并训练1D CNN模型
    :param features: 特征矩阵 (n_samples, n_features)
    :param labels: 标签列表
    :param num_classes: 类别数
    :param epochs: 训练轮数
    :param batch_size: 批大小
    :return: 训练好的模型、训练历史
    """
    # 1. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if num_classes is None:
        num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=num_classes)
    # 2. 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(features, y_cat, test_size=0.2, random_state=42)
    # 3. 构建1D CNN模型
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 4. 训练模型
    history = model.fit(
        X_train[..., np.newaxis], y_train,
        validation_data=(X_val[..., np.newaxis], y_val),
        epochs=epochs, batch_size=batch_size, verbose=2
    )
    return model, history