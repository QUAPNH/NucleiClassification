# NucleiClassification
The code of a  scale and region-enhanced decoding network for nuclei classification.[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809423000599?via%3Dihub)


As part of this repository, we supply model weights trained on the following datasets: [Google drive](https://drive.google.com/drive/folders/1J_MLYH3cW2119ZVxGcXoCq_TOEsJuge0?usp=share_link)


Installation
======
This framework implementation is based on [HoverNet](https://github.com/vqdang/hover_net). Therefore the installation is the same as original HoverNet.

Data preparation
======
    ../train/256x256_256x256/

    		img_Adrenal_gland_1_01171_000.npy

    		img_Adrenal_gland_1_01172_000.npy

    		...

    		img_Uterus_1_02591_000.npy

    ../train/Images/

    		img_Adrenal_gland_1_01171_000.png

    		img_Adrenal_gland_1_01172_000.png

    		......

    		img_Uterus_1_02591_000.png

    ../train/Labels/

    		img_Adrenal_gland_1_01171_000.mat

    		img_Adrenal_gland_1_01172_000.mat

    		......

    		img_Uterus_1_02591_000.mat

    ../train/Masks/

    		img_Adrenal_gland_1_01171_000/

    			0.png
				1.png
				......
				N.png

    		img_Adrenal_gland_1_01172_000/

				0.png
				1.png
				......
				N.png
                
    		......	
            
    		img_Uterus_1_02591_000/

				0.png
				1.png
				......
				N.png

Acknowledgments
======
We would like to thank [HoverNet](https://github.com/vqdang/hover_net) for overall framework.

Reference
======
    @article{xiao2023scale,
      title={A scale and region-enhanced decoding network for nuclei classification in histology image},
      author={Xiao, Shuomin and Qu, Aiping and Zhong, Haiqin and He, Penghui},
      journal={Biomedical Signal Processing and Control},
      volume={83},
      pages={104626},
      year={2023},
      publisher={Elsevier}
    }
