# Video Super Resolution 
## Installation guide
### Setup back-end

 1.  Install [**anaconda**](https://docs.anaconda.com/anaconda/install/) 
 2. Install the Graduation Project environment
2.1. Navigate to `Code > BackEnd >  Video-Super-Resolution`
2.2. Open terminal and execute the following command
`conda create -n <environment-name> --file req.txt`
 3. Install [**PyCharm**](https://www.jetbrains.com/pycharm/download/)
 4. Open project in PyCharm and setup the environment that is the recently created in `req.txt`
 5. Run Server
	1. Navigate to `Code > BackEnd >  Video-Super-Resolution > DeployML`
	2. Run the following command
	`python manage.py runserver`
 6. You can access the server that has started development at http://127.0.0.1:8000/
 7. Run API to upscale the video using this
	post api -> http://127.0.0.1:8000/model/
	body video -> video file, userId -> string

	the reuslts is the video path in folder

### Setup Front-end

 1. Install [**Android Studio**](https://developer.android.com/studio/install)
 2. Install [**Flutter**](https://flutter.dev/docs/get-started/install)
 3. Open Android Studio
 4. Open project in Android Studio by navigating to `Code > FrontEnd > super_video`
 5. Run the mobile application
