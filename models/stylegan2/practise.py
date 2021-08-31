from functools import partial
import numpy as np

num_init_filters=1
network_capacity=16
num_layers = int(np.log2(2048) - 1)
fmap_max=2048

filters = [network_capacity * (2 ** i) for i in range(num_layers)][::-1]
print(filters)

set_fmap_max = partial(min, fmap_max)
filters = list(map(set_fmap_max, filters))

in_out_pairs = list(zip(filters[:-1], filters[1:]))
# filters[-1] = filters[-2]
print(filters)

# chan_in_out = list(zip(filters[:-1], filters[1:]))
# chan_in_out = list(map(list, chan_in_out))
# print(chan_in_out)
#
# last_chan = filters[-1]
# dec_chan_in_out = chan_in_out[:-1][::-1]
# print(dec_chan_in_out)