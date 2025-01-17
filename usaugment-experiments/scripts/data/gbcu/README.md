# Gallbladder Cancer Ultrasound (GBCU) Dataset

The Gallbladder Cancer Ultrasound (GBCU) dataset contains a total of 1255 (432 normal, 558 benign, and 265 malignant) annotated abdominal Ultrasound images collected from 218 patients. Of the 218 patients, 71, 100, and 47 were from the normal, benign, and malignant classes, respectively.

The sizes of the training and testing sets are 1133 and 122, respectively. To ensure generalization to unseen patients, all images of any particular patient were either in the train or the test split. The number of normal, benign, and malignant samples in the train and test set is 401, 509, 223, and 31, 49, and 42, respectively. The width of the images is between 801 and 1556 pixels, and the height is between 564 and 947 pixels due to the cropping of patient-related information. Grayscale B-mode static images, including both sagittal and axial sections, were recorded by radiologists for each patient using a Logiq S8 machine.

([Paper](https://ieeexplore.ieee.org/document/9879895), [Dataset](https://gbc-iitd.github.io/data/gbcu))