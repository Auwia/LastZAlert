#!/bin/bash

# IP del dispositivo (fisso o dinamico)
DEVICE_IP="192.168.0.95"
PORT="5555"

# Avvia adb server se necessario
adb start-server

# Connetti al dispositivo
adb connect $DEVICE_IP:$PORT

