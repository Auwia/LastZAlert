adb shell "ls -t /sdcard/DCIM/Screenshots | head -n 1" | \
tr -d '\r' | \
xargs -I {} adb pull "/sdcard/DCIM/Screenshots/{}" AndroidScreenshots

