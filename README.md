The data needs to be placed in the "data" directory.

# Sources for data

**The depression data**
[Depression_detection_using_Twitter_post](https://github.com/eddieir/Depression_detection_using_Twitter_post/blob/master/depressive_tweets_processed.csv)

**The happy data**
[Emotions in text](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text)


# Requirements
    matplotlib
    numpy
    pandas
    PyQt5
    sklearn

# Possible errors:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.

Abbort (SIGABRT)
```

**Try this**
```
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
```
