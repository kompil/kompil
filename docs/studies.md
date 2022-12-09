# NN parameters reduction
* Reduce input layer : Gray code ?
* Global : Sparsing algo ?
* Reduce output layer : Deconv  ?

# File size reduction
* Quantization
* Sparse++
* Own storage format (+ compress ?)
* Weights stored as dict : value -> list of idx
* PCA over stored weights ?

# Better learning
* Better loss func
    * MS SSIM
    * SSIM
    * Pixel wise
    * Gram criterion ==> Texture loss
    * Alternate loss ?
    * Total variation
    * Euclidian
    * Gaussian kernel
    * Bound PSNR avoid overfit best frame
    * BPSNR with dynamic bound based if a min PSNR value is reached
    * Penalise to good frame (high PSNR)
* Algo to find NN hyper parameters
* Activations :
    * ReLU
    * LReLU
    * PReLU
* Custom optimizer
* 3d (temporal) sequence input
* Motion data
* Dense layer heuristic based on movie structure
    * By now, better results if more nodes in dense layers
    * Commly used : 1000 nodes by layer    
* Detect dead weight, modify them randomly ?
* Wavelet transformation
* Avg/max pool to get principal image feature ?
* How to overfit neural network ?
    * Avoid batchnorm ==> change init weight instead
    * Avoid dropout ==> maybe use later to reduce nn weight
    * Maybe small LR with / lot of epoch / harsh loss ?
    * Biais ?
    * Change input frames (saturation, channel system...) to be more "confortable for learning" ?
    * Progressive learning over resolution ? Learn x8, then learn x4, then x2, then full resolution
    * Reduce batch size ?

# Speed up learning
* Base LR
* Dynamic LR (scheduler)
* Higher batch size
* Robin hood
* Fast compute topologies
* Better init weights

# SR
* SR by video ?
* Global SR system ? -> reduce weight for each video
* Use many frame resolution during learning
* Target 1080p, then upscale (SR) or downscale if need

# Evaluation
* Better indicative performance than PSNR/SSIM
* Sliding window of X frames, storing indexes of bad frame (PSNR < fixed val), compute the std of the sliding queue; if too small, it means too many bad frames are close ==> bad encoding