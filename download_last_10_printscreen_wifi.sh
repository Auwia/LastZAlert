adb shell "ls -t /sdcard/DCIM/Screenshots | head -n 15" | \
tr -d '\r' | \
xargs -I {} adb pull "/sdcard/DCIM/Screenshots/{}" AndroidScreenshots

