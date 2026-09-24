# LastZAlert

LastZAlert is a Python automation project for **Last Z: Survival Shooter** running on an Android device.

The bot uses ADB, screenshots, OpenCV/OCR and image matching to detect game elements and execute automated workflows.

## Requirements

Tested on Ubuntu Linux with Python 3.

Install the required system packages:

    sudo apt update
    sudo apt install -y git python3 python3-full python3-venv android-tools-adb tesseract-ocr

## Installation

Clone the repository:

    git clone https://github.com/Auwia/LastZAlert.git
    cd LastZAlert

Create the environment and install dependencies:

    ./setup_env.sh

Activate the environment:

    source venv/bin/activate

## Android / ADB

Enable Developer options and USB debugging on the Android device.

Connect the phone and verify the connection:

    adb devices

The device must appear with status `device`.

Configure your Android device in `.env`:

    ADB_DEVICE=192.168.0.95:5555

Use the IP address and port assigned to your own Android device.

Wireless ADB can also be used if it is already configured on the Android device.

## Discord notifications

Discord notifications are optional.

The setup script creates `.env` from `.env.example`.

Configure your webhook only inside `.env`:

    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

**Never put a real webhook URL directly in the source code or commit it to Git.**

The `.env` file is ignored by Git.

## Security

The repository includes a pre-commit security hook that checks staged changes for known secret patterns, including Discord webhook URLs and private keys.

The setup script enables it automatically.

## Run LastZAlert

Start the game on the Android device, then:

    ./start.sh

The launcher automatically starts:

- AndroidTouchGrab to block physical touch input while keeping ADB control available
- the LastZAlert web monitor
- the LastZAlert automation bot

This is especially useful when controlling a phone with a damaged or unusable touchscreen.

Stop everything with `Ctrl+C`.

## Updating

Pull the latest version and update dependencies:

    git pull
    source venv/bin/activate
    python -m pip install -r requirements.txt

## License

This project is licensed under the MIT License. See `LICENSE` for details.
