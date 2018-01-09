import yaml

class Camera():
    def __init__(self):
        f = open('/home/yxiao1996/.ros/camera_info/head_camera.yaml')
        self.config = yaml.load(f)
        self.w = self.config["image_width"]
        self.h = self.config["image_height"]
        self.camera_matrix = self.config["camera_matrix"]
        self.dist_coeff = self.config["distortion_coefficients"]
        self.R = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
        self.t = [0]
if __name__ == '__main__':
    c = Camera()
    print c.w, c.h, c.camera_matrix, c.dist_coeff