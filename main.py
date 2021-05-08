import cv2
from purethermal2_pymodule.pt2_api import PyPureThermal2

SCALE = 4.0


def main():
    pt2_camera = PyPureThermal2()
    while True:
        if pt2_camera.update():
            thermal_image = pt2_camera.thermal_image_colorized.copy()
            thermal_image_scaled = cv2.resize(thermal_image, None, fx=SCALE, fy=SCALE)
            cv2.imshow("thermal", thermal_image_scaled)
            print(f"Cmax: {pt2_camera.thermal_image_cercius.max()}")
        key = cv2.waitKey(10)

        if key & 0xFF in [ord("q"), 27]:
            break
        if key & 0xFF == ord("s"):
            if "thermal_image_scaled" in locals():
                print("saved")
                cv2.imwrite("thermal.png", thermal_image_scaled)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
