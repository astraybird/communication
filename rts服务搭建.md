### rts服务搭建

1 下载easydarwin包

https://github.com/EasyDarwin/EasyDarwin/releases

2 解压执行 start.bat

3 在linux执行

```
ffmpeg -re -i /data4/huang/data/wuhe_zhangheng_E_9-00_9-30.mp4 -vcodec copy -acodec copy -f rtsp rtsp://9.91.180.21:554/wuhe_zhangheng_e.mp4
```

