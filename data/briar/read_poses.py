#!/usr/bin/env python
import numpy
from pprint import pp


vals = numpy.load('../nerd/moldGoldCape_rescaled/poses_bounds.npy', )

vals = vals[0:101]

pp(vals)
print(len(vals))

numpy.save('briar/poses_bounds.npy', vals)