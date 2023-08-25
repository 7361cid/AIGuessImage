import numpy as np

from keras.datasets import mnist
from PIL import Image

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

""" Network Parameters """
num_labels = 10
batch_size = 10
hidden_size = 50
iter = 5


def img_to_array(img_path, resize=True, convert_l=False):
    if resize:
        img = Image.open(img_path).resize((28, 28))
    else:
        img = Image.open(img_path)
    if convert_l:
        img = img.convert("L")
    img_arr = np.array(img, dtype='float32')
    return img_arr


def img_from_array(np_arr):
    new_img = Image.fromarray(np_arr)  # mode="L"
    return new_img


def softmax(x, axis=1):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=axis, keepdims=True)


class Net:
    def __init__(self):
        self.weights = 2 * np.random.random((784, num_labels)) - 1

    def feedforfard(self, input, axis=1):
        out = softmax(np.dot(input, self.weights), axis)
        return out

    def change_weights(self, diff, input_data):
        self.weights = self.weights + (input_data.T.dot(diff) / batch_size)

    def predict_img_by_path(self, path):
        array_from_img = img_to_array(path, convert_l=True)
        data_from_img = array_from_img.reshape(1, 28 * 28) / 255
        predict = self.feedforfard(data_from_img[0], axis=0)
        predic_result = np.argmax(predict)
        return predic_result


images = train_X[0:1000].reshape(1000, 28 * 28) / 255  # на 10К ОШИБКИ
labels = train_y[0:1000]
new_labels = np.zeros((len(labels), 10))

for i, j in enumerate(labels):
    new_labels[i][j] = 1
labels = new_labels
net_obj = Net()

for i in range(iter):   # Network learn
    error = 0
    correct_count = 0
    all_count = 0
    for j in range(int(len(images) / batch_size)):
        input_data = images[j * batch_size:(j + 1) * batch_size]
        labels_for_batch = labels[j * batch_size:(j + 1) * batch_size]
        predict = net_obj.feedforfard(input_data)
        error += np.sum((labels_for_batch - predict) ** 2)

        for k in range(batch_size):
            if np.argmax(predict[k]) == np.argmax(labels_for_batch[k]):
                correct_count += 1
            all_count += 1
            diff = labels_for_batch - predict
            net_obj.change_weights(diff, input_data)

    print(f"Network learn progress iter{i}  error {error}  correct_count {correct_count}  all_count {all_count}")


if __name__ == "__main__":
    images_test = test_X[0:10].reshape(10, 28 * 28) / 255  # проверить на train
    labels_test = test_y[0:10]

    correct_count_test = 0
    data_from_lib = images_test[0]
    predict = net_obj.feedforfard(data_from_lib, axis=0)

    for i in range(10):
        img_path = fr"C:\Users\chern\PycharmProjects\jsPractice\net\image_test{i}.jpg"
        array_from_img = img_to_array(img_path)

        img_from_array(test_X[i]).save(f"images_for_tests/image_test{i}.jpg")

        data_from_img = array_from_img.reshape(1, 28 * 28) / 255
        predict = net_obj.feedforfard(data_from_img[0], axis=0)
        predic_result = np.argmax(predict)
        if predic_result == labels_test[i]:
            correct_count_test += 1
        print(f"predic_result from img {predic_result} - {labels_test[i]}")
    print(f"TEST RESULT {correct_count_test} from 10  iter {iter}")
    # FOR 1 layer, 5 iter, train_X[0:1000]  7/10  6/10 8/10 6/10 7/10
    # FOR 1 layer, 10 iter, train_X[0:1000]  8/10  7/10 7/10 8/10 8/10
    # FOR 1 layer, 15 iter, train_X[0:1000]  8/10  8/10 8/10 8/10 8/10
    
