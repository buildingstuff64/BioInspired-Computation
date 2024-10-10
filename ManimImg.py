import tempfile
import numpy as np
from manim import *
from manim.utils.file_ops import open_media_file
from pygments.styles.rainbow_dash import GREY_DARK


class manimImage(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)
        self.play(Create(circle))
        self.wait(1)
        self.remove(circle)
        self.wait(1)
        g = self.NN([6, 5, 7, 3])
        self.play(Create(g))
        self.wait(5)

    def NN(self, layers):
        edges = []
        partitions = []
        c = 0

        for i in layers:
            partitions.append(list(range(c + 1, c + i + 1)))
            c += i
        for i, v in enumerate(layers[1:]):
            last = sum(layers[:i + 1])
            for j in range(v):
                for k in range(last - layers[i], last):
                    edges.append((k + 1, j + last + 1))

        vertices = np.arange(1, sum(layers) + 1)

        graph = Graph(
            vertices,
            edges,
            layout = 'partite',
            partitions = partitions,
            layout_scale = (6, 3),
            vertex_config = {'radius': 0.20},
            edge_config = {'color' : GREY_DARK}
        )
        return graph


with tempconfig({'save_last_frame' : True}):
    print("")
img = manimImage()
img.render()

