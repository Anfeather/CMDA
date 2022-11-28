#Class-COCO. A popular multimodal dataset MS COCO contains images and the corresponding sentences annotated by Amazon Mechanical Turk. To run classification and anomaly detection tasks on MS COCO, we propose Class-COCO. The training set is grouped according to the objects in the images. Besides, we drop images that contain two or more objects. There are 126,055 samples divided into 80 groups. For image anomaly detection tasks, we select groups with a sample size greater than 6,000 as normal samples (there are 6 groups, totaling 45,205), while 15,150 samples of 50 groups are regarded as abnormal samples. As for text anomaly detection, there are 30 groups, totaling 110,905 normal samples, while 15,150 samples of 50 groups are regarded as abnormal samples. Dataset of classification tasks is divided in the same way.
	
#Wikipedia contains 2,866 image-text pairs that belong to 10 classes. For anomaly detection, we divide the dataset into normal samples (3 classes,  totaling 921) and abnormal samples (7 classes, totaling 1,945). As for classification, we divide the dataset into 2 subsets: 2,273 and 593 pairs for training and testing, respectively. Note that we follow \cite{hu2021learning} to use the precomputed Wikipedia as input.
	
#Oxford-102 dataset of flower images contains 8,189 pairs of flowers from 102 different categories. For anomaly detection, we divide the dataset into normal samples (30 classes,  totaling 4,018) and abnormal samples (72 classes, totaling 4,171). As for classification, we divide the ten classes with the largest amount of data into 2 subsets: 1,861 and 442 pairs for training and testing, respectively.
  
# Class-COCO
for vision modality
python /home/ray/preject/CMDA/CMDA/train_image_with_CMA.py

for language modality
/home/ray/preject/CMDA/CMDA/train_text_CMDA.py

data root: ./non_iid_MSCOCO_train_30_50_5images/

Class-COCO: https://drive.google.com/file/d/1aWos8ztu8w2hevZHll2H9H5O7Pn9ThrT/view?usp=sharing
