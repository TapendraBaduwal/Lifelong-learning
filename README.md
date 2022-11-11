# learning


## Publishing python code to Pypi package

1. Create accout in [pypi](https://pypi.org/)
2. Generate API tokens from (https://pypi.org/manage/account/token/) and add this token to Github (Repository name/Setting/Sectres/Action Secrets/New reposatory secret/(python-package.yml file ko  password = secrets.test_pypi_password  bata test_pypi_password lai name ma rakhne ra API tokens lai value ma rakhera Add secrfets  ma click garne.) 
API tokens provide an alternative way to authenticate when uploading packages to PyPI.

3. Properly capture dependencies in requirements.txt file
    
    **python3 -m venv kvenv**
    
    or
    
    **pipenv --python 3.8**
    
    
    **source kvenv/bin/activate**
    
    or 
    
    **pipenv shell**

    **venv/bin/python3 -m pip freeze > requirements.txt**
    
    or 
    
    **pip freeze > requirements.txt**

    **venv/bin/python3 -m pip install -r requirements.txt**
    
    or
    
    **pip install -r requirements.txt**

    **check the venv lib folder sitepackages to know the capture dependencies**


4. Setup CI/CD pipeline for automatic deployment from github to pypi:
Creats [python-package.yml] file(.github folders vitra Workflows folder creat garne tesvitra pthon-package.yml file creats garera tesma worksflows ko rules lekhne) mainly focused on (password: ${{ secrets.test_pypi_password }},repository_url: https://upload.pypi.org/legacy/).
[link1](https://www.section.io/engineering-education/setting-up-cicd-for-python-packages-using-github-actions/), [Link2](https://py-pkgs.org/08-ci-cd.html)

5. Make Packages  jun folder lai hamile packages bauna xa (src vane folder banyem  tesvitra arko abc vanne folder banayera teslai packages bauna abc vitra __init__.py file creat garna parxa jasle abc folder aba packages ho vanne januxa).
Note: Jun jun kura haru like project ko project.py files ra teyo sanga related vako files haru like input.json, train model haru sabai abc packages vitra nai huna parxa).
Path haru ramro sanga setup garnu parne hunxa python.py files ma.
Note: Non-python files aharu packages ma adda garne tarika package_data={'': ['inputraw.json','ocrmodel']}


6. Creat [setup.py] python files in which we should writes some commands, Focus on requiremnts.txt commands installation, hamile banako packages ko directory, Python files packages banuda nai  setup hunxa ra non-python files haru lai ni xuttai add garnu loarne hunxa.
Path haru ramro sanga setup garnu parne hunxa setup.py files ma.
If hamle README.md ,LICENSE files haru ni add garna xa vane setup.py file bata add garne.

7. Creat .gitignore file ,file which are not required to push on github should be ignored vfrom this files,writs fine names inside this file to ignogres such files like venv/, __pycache__/.

8. Finally make git init,git add .,git push -m "messages",git push.
  Look at  github and go through reposatory and select Action and click on latest workflow to know all workings setups.
  
   Note: while push new verion on  github  and pypi.org, changes the name and version og our packages from setup.py files and delet the old packages     from pypi.org accounts.

9. If our Deployment completed then go through the [https://pypi.org/manage/projects/] and view  pip install command.
   Copy that command and test on local vs-code with short user input and output code we made.
   **from packages.codefile import classname**.
   
 10. If we want to install our new veersion packages  more then one times then delet old version packages which are stores in [home/hidden files/.local/lib/python3.8/sitespackages/ our install packages(delete if we want to install new version)].
 
   

## Ubuntu Installation

 1. Download ubuntu from this link (https://releases.ubuntu.com/20.04/) annd (https://ubuntu.com/tutorials/create-a-usb-stick-on-ubuntu#1-overview)
 2. Also download or install **Startup Disk Creator**(check in ubuntu software there may be otherwise download or install it).
 3. Inster Pen Drive in local machine and open **Startup Disk Creator** and write image on pendrive disk and  make completed.
 4. After that press power off button of laptop and inster bootable Pendrvie and press **F12** button and follow the rules.
 
 
 ## Run API and Docker(django rest framework api)

       ** whole api project work under virtual env** 

       **pip install djangorestframework**(api project creat garne**

       **pip install django-cors-headers**(Make api use by all origin)
       
       **pip install python-decouple** (Hinde garne kura .env file ma rakhne using decouple)
       
       **pip install gunicorn**
       
       ** python manage.py collectstatic**( make static api)
       

        **sudo docker-compose up**
 
 
 ## Dockerfile
 
 1. sudo apt install docker.io
 
 2. sudo apt install docker-compose
 
 3.  docker --version
 
 4. docker-compose --version
 
 5. make **Dockerfile** file and write command like
 
         FROM python:3.8  

         ENV PYTHONUNBUFFRED=1

         WORKDIR /foldername

         COPY requirements.txt .

         RUN pip install -r requirements.txt 

         COPY . .

         COPY ./entrypoint.sh /

         ENTRYPOINT ["sh","/entrypoint.sh"]

6. make **docker-compose.yml** file and write command like 

         version: '3.3'

         services:

           dockerimagename:

             restart: always

             image: dockerimagename


             build: .
          
          
  7. make **entrypoint.sh** file and write command like

         #!/bin/sh

         python3 -m resultkpd
         
         or
         
         for single user support
        
         python manage.py runserver 0.0.0.0:8003
         
         or

         Support multi user
        
         gunicorn kapediamlapis.wsgi:application --bind 0.0.0.0:8003
         
         or
         
         gunicorn kapediamlapis.wsgi:application --bind 0.0.0.0:8003 --timeout 600


 
 8. **sudo docker-compose up** ==Compose docker file
 
 9. **sudo docker images** == To know the blackbox of docker
 
 10. **Note:: if any change in docker files code rebuilt docker images**
      
       **sudo docker ps -a** == Check the container id
       
       **sudo docker stop  6de040f0cefc** ==Stop Container id at first
       
       **sudo docker rm 6de040f0cefc** == Remove container id  **or**
       **docker rm -f 6de040f0cefc** == Forces the removal of a running container
      
       **sudo docker images** == Check the docker image id 
       
       **sudo docker rmi 6ad660682d7f --force** ==Remove and delete docker images with its id
       
       **sudo docker-compose up**==Re Builts Docker images
 
 11. **Note: if any erro in Dockerfile by**
 
      RUN apt-get update && apt-get install -y python3-opencv

      RUN pip install opencv-python
      
      Solution: 
      **sudo apt-get update**
      
      **sudo docker builder prune , sudo docker system prune --volumes**
 
     I had the same issue. For me the issue was that my "apt update -y" step was cached and thus contained wrong repo's. Fixed it by forcing it to not    use cache. To clear the cache before run use **docker builder prune** or if that doesn't work (as it didn't for me) try something more aggressive such as **docker system prune --all**



 ## PULC Classification Model of Language
 
1. Documentation(https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/PULC/PULC_language_classification_en.md)

2. Quick Start
 
        python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

        pip3 install paddleclas

        Note: paddleclas model accept image array in RGB i.e 3 channel img array.
        
        from paddleclas import PaddleClas
        clas = PaddleClas(model_name="language_classification", inference_model_dir= self.lang_classification_dir_path)
        infer_imgs = '/home/tapendra/Desktop/kapediaml/images/docnp (1).png'
        result=clas.predict(infer_imgs)
        print(next(result))
        
   

 3. Research Paper

       PP-LCNet: A Lightweight CPU Convolutional Neural Network(https://arxiv.org/abs/2109.15099)
       
    
    
## PPOCRLabelv2
 1. Documentation(https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/PPOCRLabel)

 2. Installation and Run:
 
        pip3 install --upgrade pip

        python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple


        Ubuntu Linux::
        pip3 install PPOCRLabel
        pip3 install trash-cli

        # Select label mode and run 
        PPOCRLabel  # [Normal mode] for [detection + recognition] labeling

        Error message::
        pip install opencv-python==4.2.0.32
        pyrcc5 -o libs/resources.py resources.qrc
        pip install opencv-contrib-python-headless==4.2.0.32

## Use custom model
1. Documentation(https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/whl_en.md#31-use-by-code)

2. When the built-in model cannot meet the needs, you need to use your own trained model. First, refer to the first section of inference_en.md to convert    your det and rec model to inference model(https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/inference_en.md), and then use it as   follows:
        
3. Paddleocr worked better on GRAY or BGR img or numpy array  
        from paddleocr import PaddleOCR,draw_ocr
        # The path of detection and recognition model must contain model and params files
        ocr = PaddleOCR(det_model_dir='{your_det_model_dir}', rec_model_dir='{your_rec_model_dir}', rec_char_dict_path='{your_rec_char_dict_path}',     cls_model_dir='{your_cls_model_dir}', use_angle_cls=True)
        
        img = cv2.imread(img_path) #Support numpy arrray in BGR format also if our model already support GRAY otherwise used GRAY IMG
        
       # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), If your own training model supports grayscale images, you can uncomment this line

        result = ocr.ocr(img_path, cls=True)
        for line in result:
            print(line)

        # draw result
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save('result.jpg')
        
  
  ## Images Read

When the image file is read with the OpenCV function imread(), the order of colors is BGR (blue, green, red). 

On the other hand, in Pillow, the order of colors is assumed to be RGB (red, green, blue).

1. cv2.imread() ==== always on BGR format

2. pill img mostly === RGB and other mode format

3. Paddleocr better worked on GRAY and BGR.

4. paddlecls used 3 channel RGB img.

## Handel Pdf,Images and iterate pages
       
       pdf_or_img = path
       numpyarray_img_list = []
        def pdf_to_img(self):
         try:
            for i in range(5):
                doc = fitz.open(self.input_pdfimg_file_path)
                page = doc.load_page(i)
                zoom_x = 2.5
                zoom_y = 2.5
                mat = fitz.Matrix(zoom_x, zoom_y)
                pix = page.get_pixmap(matrix = mat, alpha =False)
                #pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                numpy_image = np.asarray(img)
                #paddleocr worked better on GRAY or BGR image
                gray_numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
                numpyarray_images_list.append(gray_numpy_image)
        except ValueError:
            pass
                return numpyarray_img_list
                
        
 **Only BBOX detection**
 
                from paddleocr import PaddleOCR,draw_ocr
                from PIL import Image

                ocr = PaddleOCR(use_angle_cls=True, lang='ne')
                img_path = '/home/tapendra/Desktop/kapediaml/images/3lineimg.png'
                result = ocr.ocr(img_path,cls=True,rec=False)
                for line in result:
                    print(line)


                image = Image.open(img_path).convert('RGB')
                im_show = draw_ocr(image, result, txts=None, scores=None, font_path='/home/tapendra/Desktop/kapediaml/nepali.ttf')
                im_show = Image.fromarray(im_show)
                im_show.save('result.jpg')

        
## Crop an image given the coordinates

1. Documentation(https://splunktool.com/how-do-i-crop-an-image-given-the-coordinates-of-the-four-corners-to-crop)

2. BBOX:
        x1, y1: 1112 711
        x2, y2: 1328 698
        x3, y3: 1330 749
        x4, y4: 1115 761

        top_left_x = min([x1, x2, x3, x4])
        top_left_y = min([y1, y2, y3, y4])
        bot_right_x = max([x1, x2, x3, x4])
        bot_right_y = max([y1, y2, y3, y4])

        img[top_left_y: bot_right_y, top_left_x: bot_right_x]
        img[top_left_y: bot_right_y + 1, top_left_x: bot_right_x + 1]


## sorting_algorthim_for_bounding_box_from_left_to_right_and_top_to_bottom

1. Documentation1(https://vigneshgig.medium.com/bounding-box-sorting-algorithm-for-text-detection-and-object-detection-from-left-to-right-and-top-cf2c523c8a85)

2. Documentation2(https://github.com/vigneshgig/sorting_algorthim_for_bounding_box_from_left_to_right_and_top_to_bottom)

3. Documentation3(https://stackoverflow.com/questions/58903071/i-want-to-sort-the-words-extracted-from-image-in-order-of-their-occurence-using)

3. Code

        def bounding_box_sorting(boxes):
            num_boxes = len(boxes)
            # sort from top to bottom and left to right
            sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
            _boxes = list(sorted_boxes)
            # print('::::::::::::::::::::::::::testing')

            # check if the next neighgour box x coordinates is greater then the current box x coordinates if not swap them.
            # repeat the swaping process to a threshold iteration and also select the threshold 
            Threshold value = ((ymax- ymin)/2) + C
            for i in range(25):
              for i in range(num_boxes - 1):
                  if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < threshold_value_y and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                      tmp = _boxes[i]
                      _boxes[i] = _boxes[i + 1]
                      _boxes[i + 1] = tmp

            print(sorted_boxes)
            return _boxes


## Add single line Bbox img array into single list

            result_tapendra = []
            temp_tapendra = []
            for index, box in enumerate(_boxes):
                current = int(box[0][1])
                try:
                    next = int(_boxes[index+1][0][1])
                except IndexError:
                    result_tapendra.append(temp_tapendra)

                threshold_value_y = ((bot_right_y- top_left_y)/2) + 2
                if (next_top_left_y1 - current_top_left_y1) <= threshold_value_y:
                    temp_tapendra.append(current)

                else:
                    temp_tapendra.append(current)
                    result_tapendra.append(temp_tapendra)
                    temp_tapendra = []
            print(result_tapendra)
            

            
## How to Increase Image Resolution

1. Doc(https://buildmedia.readthedocs.org/media/pdf/pymupdf/latest/pymupdf.pdf)

          for i in range(10):
            doc = fitz.open(input_imgfile_path)
            page = doc.load_page(i)
            zoom_x = 2.0 # horizontal zoom
            zoom_y = 2.0 # vertical zoom
            mat = fitz.Matrix(zoom_x, zoom_y) # zoom factor 2 in each dimension
            pix = page.get_pixmap(matrix=mat) # use 'mat' instead of the identity matrix

       or 
       dpi concept ======dots per inch can be used in place of "matrix" dpi value is saved with the image file – 
       which does not happen automatically when  using the Matrix notation.
       pix = page.get_pixmap(dpi=300)
                

## Some GIT commands                                                 
                                                       
   1. To check the git status
   
          git status
          
   2. To create new branch 

          git branch <branch_name>
          
   3. To add all changes code in stating area 

          git add . 
          
   4. To and specific file in stating area

          git add <file_name>
          
   5. To commit git files

          git commit -m "your commit message"
          
          
   6. To push into github accout 

          git push origin <branch_name_where_you_wanna_push_code>
   
   7. To check git log and to get commit id

          git log

       
   8. To rest git staus
   
          git reset 
          
   9. To rest code upto specfic commit id

          git reset <your_commit_id>
          
  10. To pull code from the specific branch

          git pull origin <branch_name>
   
   11. To git reset hard (it will ignore all the code which are not committed so be careful before using this)

            git reset --hard
          
       
## Multilingual OCR Development Plan

1. Documentation(https://github.com/PaddlePaddle/PaddleOCR/issues/1048)

2. We can download single model from here.


## PaddleOCR rec model training


1. Doc(https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_en/recognition_en.md)


 a. **ppocr/utils/dict/devanagari_dict.txt  setup garne jasma lang ko words haru hunxan.**

 b. PPOCRlabel bata label gareko data lai asari milaune...train_data as folder ho

 
 Train:
             -train_data
                |- rec_gt_train.txt
                |- train/crop_img
                    |- word_001.png
                    |- word_002.jpg
                    |- word_003.jpg
                    | ...
                    
          
          
          rec_gt_train.txt  ko format as below huna parxa
   
                img1_crop_0.jpg	अगाडि,
                img1_crop_1.jpg	अझै,
                img1_crop_2.jpg	अनुसार,
                img1_crop_3.jpg	अन्तर्गत,
                img1_crop_4.jpg	अन्य,
                img1_crop_5.jpg	अब,
                
Test:
            -train_data
                |- rec_gt_test.txt
                |- train/crop_img
                    |- word_001.png
                    |- word_002.jpg
                    |- word_003.jpg
                    | ...
                    
          
          
          rec_gt_test.txt  ko format as below huna parxa
   
                img1_crop_0.jpg	अगाडि,
                img1_crop_1.jpg	अझै,
                img1_crop_2.jpg	अनुसार,
                img1_crop_3.jpg	अन्तर्गत,
                img1_crop_4.jpg	अन्य,
                img1_crop_5.jpg	अब,
                
c.**devanagari_PP-OCRv3_rec.yml** path setup **PaddleOCR/configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml**

        Changes file like this::

        use_gpu: false .....if no GPU


        save_model_dir: ./output/rec

        infer_img: /home/tapendra/Desktop/PaddleOCR/images/test.png

        character_dict_path: ppocr/utils/dict/devanagari_dict.txt

        save_res_path: ./output/predicts_ppocrv3_devanagari.txt


        Train::

        data_dir: ./train_data/rec/train

        label_file_list: ["./train_data/rec/rec_gt_train.txt"]

        EVL/Test::

         data_dir: ./train_data/rec/test

         label_file_list: ["./train_data/rec/rec_gt_test.txt"]


                
**You can also train PaddleOCR on CPU.**

d. You need to install the CPU version of paddle first.

         python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

e. Set the use_gpu parameter to False

 ## For training/evaluation/prediction:

    f. python3 tools/train.py -c configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml -o Global.use_gpu=True

 ## For predict:

    g. python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/devanagari_det_mv3_db/" --  rec_model_dir="./inference/devanagari_rec_mv3_crnn/" --use_gpu=True

    step5. After training following parameters will be saved in output_model/rec folder:

                output_model/rec/
                ├── best_accuracy.pdopt
                ├── best_accuracy.pdparams
                ├── best_accuracy.states
                ├── config.yml
                ├── iter_epoch_3.pdopt
                ├── iter_epoch_3.pdparams
                ├── iter_epoch_3.states
                ├── latest.pdopt
                ├── latest.pdparams
                ├── latest.states
                └── train.log


    h. Convert trained model to inference model:

           # Global.save_inference_dir Set the address where the converted model will be saved.


            python3 tools/export_model.py -c configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml -o Global.pretrained_model=devanagari_PP-OCRv3_rec_train/best_accuracy  Global.save_inference_dir=./inference/devanagari_PP-OCRv3_rec/ --use_gpu=True

    i. After the conversion is successful, there are three files in the model save directory:

                inference/devanagari_PP-OCRv3_rec/
                    ├── inference.pdiparams         # The parameter file of recognition inference model
                    ├── inference.pdiparams.info    # The parameter information of recognition inference model, which can be ignored
                    └── inference.pdmodel           # The program file of recognition model




## Machine Learning
1. (https://github.com/regmi-saugat/66Days_MachineLearning)

## Table Extraction Table recognition

1. Video(https://www.youtube.com/watch?v=HZh31OGiQRQ)

2. Doc(https://colab.research.google.com/drive/1I-tp71bSdQmXG6wwqAPTd9Sn63Rtf4nC#scrollTo=w5MA08E0F8aU)

3. https://arxiv.org/pdf/2105.01848.pdf 

4. https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppstructure/table 

5. https://www.youtube.com/watch?v=HZh31OGiQRQ 

6. https://huggingface.co/docs/transformers/main/model_doc/table-transformer

7. https://huggingface.co/models?other=table-transformer

8. https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Table%20Transformer





## Django REST Framework 
1. Doc(https://www.django-rest-framework.org/tutorial/1-serialization/#writing-regular-django-views-using-our-serializer)
2. https://www.youtube.com/watch?v=qXXC6ocTC80&list=PLbGui_ZYuhijTKyrlu-0g5GcP9nUp_HlN&index=1

## Postman
1. Doc(https://www.postman.com/)
