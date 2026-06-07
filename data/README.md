# Dataset

Download `pavia-u.mat` from the IEEE GRSS benchmark:

https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Direct link (Pavia University scene):  
http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat  
Ground truth:  
http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat

**Important**: the experiments expect a single `.mat` file at `data/pavia-u.mat`
that contains both `data` (H×W×103 float32) and `map` (H×W int) fields.
The standard download provides them as two separate files — merge them:

```python
import scipy.io, numpy as np

img = scipy.io.loadmat('PaviaU.mat')
gt  = scipy.io.loadmat('PaviaU_gt.mat')
merged = {'data': img['paviaU'].astype('float32'),
          'map':  gt['paviaU_gt'].astype('int32')}
scipy.io.savemat('data/pavia-u.mat', merged)
```

Image size: 610×340×103 bands, 9 labeled land-cover classes (0 = unlabeled).
