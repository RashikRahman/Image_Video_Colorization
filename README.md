# Image_Video_Colorization
 
Hey there this is Rashik Rahman. This is a fun project to prectice openCV. The model files are open source and very available to use. I just took the pre-trained models and implemented those with openCV. To run this project you need to do the followings.

You need to locate to the directory where you clone this repo. You can use command promt
or anaconda powershell. To locate to repo you just need to type in this command. I am using Anaconda PowerShell. Run Anaconda PowerShell as administrator then paste the following commands.

```ini
cd 'path to repo'
```

**Next** you may need to upgrade pip using the following command
```ini
python -m pip install --user --upgrade pip
```

After that type in the following command.

```ini
pip install -r requirement.txt
```

Note that to change the inputs you will need to manually give input image path in Main.py . Now the last part is you need to download the model file. As it is too large so couldn't upload it to github and gitLfs isn't working properly for me. Dowload the [caffemodel link](https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel) model. Then past this model to the **Model** folder of this repo. Now you are ready to run and explore the code.

Now to run the program use the following command

```ini
python Main.py
```

## Outputs

![]("Output/2.jpg")

![]('Output/3.jpg')

![]('Output/5.jpg')

![]('Output/6.jpg')

![]('Output/7.jpg')

![]('Output/1.PNG')

![]('Output/2.PNG')

![]('Output/3.PNG')

![]('Output/4.PNG')

![]('Output/5.PNG')