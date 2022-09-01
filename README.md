# learning

## Publishing python code to Pypi package

1. Create accout in [pypi](https://pypi.org/)
2. Generate API tokens from (https://pypi.org/manage/account/token/) and add this token to Github (Repository name/Setting/Sectres/Action Secrets/New reposatory secret/(python-package.yml file ko  password = secrets.test_pypi_password  bata test_pypi_password lai name ma rakhne ra API tokens lai value ma rakhera Add secrfets  ma click garne.) 
API tokens provide an alternative way to authenticate when uploading packages to PyPI.

3. Setup CI/CD pipeline for automatic deployment from github to pypi:
Creats [python-package.yml] file(.github folders vitra Workflows folder creat garne tesvitra pthon-package.yml file creats garera tesma worksflows ko rules lekhne) mainly focused on (password: ${{ secrets.test_pypi_password }},repository_url: https://upload.pypi.org/legacy/).
[link1](https://www.section.io/engineering-education/setting-up-cicd-for-python-packages-using-github-actions/), [Link2](https://py-pkgs.org/08-ci-cd.html)

4. Make Packages  jun folder lai hamile packages bauna xa (src vane folder banyem  tesvitra arko abc vanne folder banayera teslai packages bauna abc vitra __init__.py file creat garna parxa jasle abc folder aba packages ho vanne januxa).
Note: Jun jun kura haru like project ko project.py files ra teyo sanga related vako files haru like input.json, train model haru sabai abc packages vitra nai huna parxa).
Path haru ramro sanga setup garnu parne hunxa python.py files ma.
5.Creat [setup.py] python files in which we should writes some commands, Focus on requiremnts.txt commands installation, hamile banako packages ko directory, Python files packages banuda nai  setup hunxa ra non-python files haru lai ni xuttai add garnu loarne hunxa.
  Path haru ramro sanga setup garnu parne hunxa setup.py files ma.
  If hamle README.md ,LICENSE files haru ni add garna xa vane setup.py file bata add garne.

6. Creat .gitignore file ,file which are not required to push on github should be ignored vfrom this files,writs fine names inside this file to ignogres such files like venv/, __pycache__/.

7. Finall make git init,git add .,git push -m "messages",git push.
  Look at  github and go through reposatory and select Action and click on latest workflow to know all workings setups.
  
 8. If our Deployment completed then go through the [https://pypi.org/manage/projects/] and view  pip install command.
   Copy that command and test on local vs-code with short user input and output code we made.
   **from packages.codefile import classname**.
   
   


## Ubuntu Installation

 1. Download ubuntu from this link (https://releases.ubuntu.com/20.04/) annd (https://ubuntu.com/tutorials/create-a-usb-stick-on-ubuntu#1-overview)
 2. Also download or install **Startup Disk Creator**(check in ubuntu software there may be otherwise download or install it).
 3. Inster Pen Drive in local machine and open **Startup Disk Creator** and write image on pendrive disk and  make completed.
 4. After that press power off button of laptop and inster bootable Pendrvie and press **F12** button and follow the rules.
