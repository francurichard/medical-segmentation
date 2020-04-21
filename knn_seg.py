import sys
from create_csv import create_csv
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
	hu_filename = sys.argv[1]
	seg_filename = sys.argv[2]
	path = create_csv((hu_filename, seg_filename))
	dataset = pd.read_csv(path)
	X = dataset[['x_position', 'y_position', 'hu_val']]
	scaler = MinMaxScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	y = dataset['seg_val']
	kernel = np.ones((5, 5), np.uint8)
	classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='uniform', p=2)
	classifier.fit(X, y)
	y_pred = np.array(classifier.predict(X), dtype=np.uint8)
	y_pred = y_pred.reshape(512, 512)
	print(y_pred)
	image = y_pred
	print(image)
	print(image.shape)
	image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=1)
	image = cv.medianBlur(image, 5)
	image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=1) 	
	plt.imshow(image, cmap='gray')
	plt.show()



