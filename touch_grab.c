#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <linux/input.h>

static int fd = -1;

static void cleanup(int sig)
{
    if (fd >= 0) {
        ioctl(fd, EVIOCGRAB, 0);
        close(fd);
    }

    printf("\nTouchscreen released.\n");
    exit(0);
}

int main(int argc, char **argv)
{
    const char *device = "/dev/input/event2";

    if (argc > 1)
        device = argv[1];

    signal(SIGINT, cleanup);
    signal(SIGTERM, cleanup);

    fd = open(device, O_RDONLY);

    if (fd < 0) {
        perror("open");
        return 1;
    }

    if (ioctl(fd, EVIOCGRAB, 1) < 0) {
        perror("EVIOCGRAB");
        close(fd);
        return 1;
    }

    printf("Touchscreen grabbed: %s\n", device);
    printf("Physical touch events are now blocked.\n");
    printf("Press Ctrl+C to release.\n");

    while (1)
        pause();

    return 0;
}
