from __future__ import absolute_import, division, print_function

import itertools
import numpy as np
from dask_patternsearch.search import RightHandedSimplexStencil
from toolz import take
import gizeh
import moviepy.editor as mpy


# DB16 - DawnBringer's 16 Col Palette v1.0
# http://pixeljoint.com/forum/forum_posts.asp?TID=12795
colors = [
    [20, 12, 28],
    [68, 36, 52],
    [48, 52, 109],
    [78, 74, 78],
    [133, 76, 48],
    [52, 101, 36],
    [208, 70, 72],
    [117, 113, 97],
    [89, 125, 206],
    [210, 125, 44],
    [133, 149, 161],
    [109, 170, 44],
    [210, 170, 153],
    [109, 194, 202],
    [218, 212, 94],
    [222, 238, 214],
]
colors = [[x / 255 for x in color] for color in colors]

halving_colors = {
    -5: colors[1],
    -4: colors[1],
    -3: colors[1],
    -2: colors[1],
    -1: colors[6],
    0: colors[8],
    1: colors[11],
    2: colors[5],
    3: colors[0],
    4: colors[0],
    5: colors[0],
    6: colors[0],
    7: colors[0],
}
grid_color = colors[15]


def make_frames(frames, width, scale):
    incrementer = itertools.count()
    stencil = RightHandedSimplexStencil(2, 30)
    rotate = np.array([1, -1])
    offset = width / 2 + rotate * width / 10
    points = list(take(frames, stencil.generate_stencil_points()))
    for point in points:
        point.point = rotate * point.point * width / 12 + offset

    def make_frame(t):
        i = next(incrementer)
        surface = gizeh.Surface(width=width, height=width, bg_color=(1, 1, 1))

        line = gizeh.polyline([[offset[0], 0], [offset[0], width]], stroke=grid_color, stroke_width=2)
        line.draw(surface)
        line = gizeh.polyline([[0, offset[1]], [width, offset[1]]], stroke=grid_color, stroke_width=2)
        line.draw(surface)

        x = offset[0] + width/scale
        y = offset[1] - width/scale
        while x <= width + 1:
            line = gizeh.polyline([[x, 0], [x, width]], stroke=grid_color, stroke_width=0.5)
            line.draw(surface)
            line = gizeh.polyline([[0, y], [width, y]], stroke=grid_color, stroke_width=0.5)
            line.draw(surface)
            x += width/scale
            y -= width/scale
        x = offset[0] - width/scale
        y = offset[1] + width/scale
        while x >= -1:
            line = gizeh.polyline([[x, 0], [x, width]], stroke=grid_color, stroke_width=0.5)
            line.draw(surface)
            line = gizeh.polyline([[0, y], [width, y]], stroke=grid_color, stroke_width=0.5)
            line.draw(surface)
            x -= width/scale
            y += width/scale

        circle = gizeh.circle(r=3.25, xy=offset, fill=halving_colors[0])
        circle.draw(surface)
        if i > 0:
            for i in range(i-1):
                point = points[i]
                color = halving_colors[point.halvings]
                circle = gizeh.circle(r=max(0.5, 3.25 - 0.75*point.halvings), xy=point.point, fill=color)
                circle.draw(surface)
        return surface.get_npimage()
    return make_frame


def make_gif(frames, fps=8, width=320, scale=11, filename='stencil.gif'):
    clip = mpy.VideoClip(make_frame=make_frames(frames, width, scale), duration=frames / fps)
    clip.write_gif(filename, fps=fps)


if __name__ == '__main__':
    make_gif(120, filename='stencil120-orig.gif')
    print('\n"stencil120-orig.gif" written.  I highly recommend optimizing it with gifsicle:\n')
    print('gifsicle --colors=256 -O2 stencil120-orig.gif -o stencil120.gif\n')

