scenes = [
    "../images/uturn_scene.jpg",
    "../images/leftturn_scene.jpg",
    "../images/rightturn_scene.jpg"
]

for filename in scenes:
    scene_img = cv2.imread(filename, 0)
    pred = tm.predict(scene_img)
    print filename.split('/')[-1]
    print pred 