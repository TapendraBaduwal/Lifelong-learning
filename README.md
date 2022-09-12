# learning
## Publishing python code to Pypi package

1. Create accout in [pypi](https://pypi.org/)
2. Generate API tokens from (https://pypi.org/manage/account/token/) and add this token to Github (Repository name/Setting/Sectres/Action Secrets/New reposatory secret/(python-package.yml file ko  password = secrets.test_pypi_password  bata test_pypi_password lai name ma rakhne ra API tokens lai value ma rakhera Add secrfets  ma click garne.) 
API tokens provide an alternative way to authenticate when uploading packages to PyPI.

3. Properly capture dependencies in requirements.txt file
    
    **python3 -m venv kvenv**
    
    **source kvenv/bin/activate**

    **venv/bin/python3 -m pip freeze > requirements.txt**

    **venv/bin/python3 -m pip install -r requirements.txt**

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
 
 
 
 ## PULC Classification Model of Language
 
1. Documentation(https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/PULC/PULC_language_classification_en.md)

2. Quick Start
 
        python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

        pip3 install paddleclas


        import paddleclas
        model = paddleclas.PaddleClas(model_name="language_classification")
        result = model.predict(input_data="pulc_demo_imgs/language_classification/word_35404.png")
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

2. When the built-in model cannot meet the needs, you need to use your own trained model. First, refer to the first section of inference_en.md to convert    your det and rec model to inference model(https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/inference_en.md), and then use it as   follows
        from paddleocr import PaddleOCR,draw_ocr
        # The path of detection and recognition model must contain model and params files
        ocr = PaddleOCR(det_model_dir='{your_det_model_dir}', rec_model_dir='{your_rec_model_dir}', rec_char_dict_path='{your_rec_char_dict_path}',      cls_model_dir='{your_cls_model_dir}', use_angle_cls=True)
        img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
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
