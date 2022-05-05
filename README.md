# look-a-like
![demonstration](https://user-images.githubusercontent.com/30828805/166903508-2f9f1a44-13a4-4ca3-8300-3fa77016ca80.png)  

A project that determines which actor the person in the photo most resembles.  

To use the dataset, I first filtered it: I excluded all photos in which more than one face was found, no faces were found, or the confidence of the found face was low. Next, I converted each photo into a set of measurements using the "VGG-Face" model, and obtained only the necessary information from the dataset (the path to the photo, the name of the actor in this photo, and the resulting set of measurements for each photo) and saved the data to the file `data.pkl` (on the dataset of the ETH Zurich University, the FILE SIZE IS APPROXIMATELY EQUAL TO 1 GB) so as not to carry out calculations every time the program is executed. All this was done by the `encode.py` file.  
In the `main.py` file, I read the already saved data from the `data.pkl` file, convert my photo into a set of measurements using the same "VGG-Face" model, and calculate the cosine similarity between the measurements of my photo and the measurements of each photo from the dataset. It remains only to get the photo and the name of the actor from the dataset, whose measurements showed the greatest similarity, and display it on the screen.  

I used a dataset compiled by the ETH Zurich University, but only from IMDB.  

For the program to work, you need to additionally download the following files that are too large for GitHub to the root directory of the project: [imdb_crop](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) (dataset: photos and metadata), [data.pkl](https://drive.google.com/file/d/1LUBWMNE-_si_snwt4i-Vm6NXc5A-8CPl/view?usp=sharing), and [vgg_face_weights.h5](https://drive.google.com/file/d/1FAb3RCVo-an4gb-4ySI01nx3JgwTSqsW/view?usp=sharing) (coefficients for the "VGG-Face" model). To run the program on your own photo, replace the `target.jpg` file with the photo, keeping the file name, and run `main.py`.  
Before running the program, check that the file hierarchy is exactly like this:  
```markup
└── look-a-like
    ├── data.pkl
    ├── encode.py
    ├── haarcascade_frontalface_default.xml
    ├── imdb_crop
    │   ├── 00
    │   ├── 01
    │   ├── ...
    │   └── imdb.mat
    ├── main.py
    ├── target.jpg
    └── vgg_face_weights.h5
```
