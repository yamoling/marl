from lle import LLE, World
import cv2


if __name__ == "__main__":
    world = World.from_file("maps/doors")
    img = world.get_image()
    cv2.imwrite("doors.png", img)
