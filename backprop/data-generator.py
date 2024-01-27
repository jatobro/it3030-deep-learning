from functions import clamp

class DataGenerator:
    def __init__(self, size=20, noise_level=0.05, relative_sizes=(0.7, 0.2, 0.1)):
        self.size = clamp(size, 10, 50)
        self.noise_level = noise_level
        self.relative_sizes = relative_sizes

    def generate_image(self, width, height):
        pass
