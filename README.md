# Machine Learning

### *From-Scratch Implementations, Adapted Models, and More*


## Dependencies


--- | pip | conda|
 --- | ----------- | ----------- |
 CPU | pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html| conda install -c conda-forge numpy sklearn scipy pytorch       |
 GPU | pip install numpy sklearn scipy torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html  | conda install -c conda-forge; numpy sklearn scipy ; conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia     |


## Usage

```python
# Supervised 
import Adaboost 
import BootstrapAggregator
import LogisticRegression 
import MLP
import MulticlassGaussian
import RandomForest
import SVM

# Unsupervised
import kmeans
import PCA
```

## License

This Repository is licensed under the [CC0 1.0 Universal (CC0 1.0)   ](https://creativecommons.org/publicdomain/zero/1.0/ )
