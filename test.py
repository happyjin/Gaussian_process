import numpy as np
ncoord = np.matrix('3225   318;2387    989;1228    2335;57      1569;2288  8138;3514   2350;7936   314;9888    4683;6901   1834;7515   8231;709   3701;1321    8881;2290   2350;5687   5034;760    9868;2378   7521;9025   5385;4819   5943;2917   9418;3928   9770')
ncoord = np.array(ncoord)
D = np.sqrt(((ncoord[:, :, None] - ncoord[:, :, None].T) ** 2).sum(1))

y = np.asarray([1,2,3,4,5,6,7,8,9]).reshape(-1,3)
a = np.asarray([1,-1,-1,1,1,1,1,-1]).reshape(-1,1)
print len(y)