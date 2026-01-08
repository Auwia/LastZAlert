import subprocess

# Cambia questo con il nome esatto del pacchetto della tua app
PACKAGE_NAME = "com.readygo.barrel.gp"

def kill_app(package_name):
    try:
        result = subprocess.run(
            ["adb", "shell", "am", "force-stop", package_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"[✓] App '{package_name}' chiusa con successo.")
        else:
            print(f"[✗] Errore nella chiusura dell'app: {result.stderr}")

    except FileNotFoundError:
        print("[!] adb non trovato. Assicurati che sia installato e nel PATH.")

if __name__ == "__main__":
    kill_app(PACKAGE_NAME)

