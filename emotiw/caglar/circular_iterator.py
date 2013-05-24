"""
    Iterator to iterate over the image, in a circular way.
"""
def circle_around(x, y):
    r = 1
    i, j = x-1, y-1
    while True:
        while i < x+r:
            i += 1
            yield r, (i, j)
        while j < y+r:
            j += 1
            yield r, (i, j)
        while i > x-r:
            i -= 1
            yield r, (i, j)
        while j > y-r:
            j -= 1
            yield r, (i, j)
        r += 1
        j -= 1
        yield r, (i, j)
