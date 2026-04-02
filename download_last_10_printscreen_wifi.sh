mkdir debug/AndroidScreenshots
adb shell "ls -t /sdcard/DCIM/Screenshots | head -n 4" | \
tr -d '\r' | \
xargs -I {} adb pull "/sdcard/DCIM/Screenshots/{}" debug/AndroidScreenshots

