# Importing Packages
from detectron2 import model_zoo
from detectron2.config import CfgNode, LazyConfig, instantiate, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects import point_rend
from copy import deepcopy
# from torchvision import models, transforms
# import torch.nn as nn
# from torch.autograd import Variable

import os
import time
import torch

# for swin-transformer----------------------------------------
# from mmcv import Config
# from mmdet.apis import init_detector

# batch predictor class.
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from omegaconf import OmegaConf


from sahi.model import Detectron2DetectionModel

class Swin_Detectron_Predictor:
   
    def __init__(self, cfg):
        """
        Args:
            cfg: a yacs CfgNode or a omegaconf dict object.
        """
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = deepcopy(cfg)
            self.model = instantiate(cfg.model)
            self.model.to(cfg.train.device)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]
            self.model.eval()
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class SahiSwinDetectron(Detectron2DetectionModel):
   
    def load_model(self):
        # check_requirements(["torch", "detectron2"])
        print("Config path: ",self.config_path,flush=True)
        cfg = LazyConfig.load(self.config_path)
        cfg.train.device = self.device
        cfg.train.init_checkpoint = self.model_path
        cfg.model.roi_heads.box_predictors[-1].test_score_thresh = self.confidence_threshold

        if self.image_size is not None:
            cfg.dataloader.test.mapper.augmentations = \
            [L(T.ResizeShortestEdge)(short_edge_length=self.image_size, max_size=self.image_size)]
        # init predictor
        model = Swin_Detectron_Predictor(cfg)

        self.model = model

        category_names = ["scratch", "dents", "tear", "clipsbroken", "shattered", "broken"]
        self.category_names = category_names
        self.category_mapping = {
            str(ind): category_name for ind, category_name in enumerate(self.category_names)
        }
        # detectron2 category mapping
        # if self.category_mapping is None:
        #     try:  # try to parse category names from metadata
        #         metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        #         category_names = metadata.thing_classes
        #         self.category_names = category_names
        #         self.category_mapping = {
        #             str(ind): category_name for ind, category_name in enumerate(self.category_names)
        #         }
        #     except Exception as e:
        #         logger.warning(e)
        #         # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
        #         if cfg.MODEL.META_ARCHITECTURE == "RetinaNet":
        #             num_categories = cfg.MODEL.RETINANET.NUM_CLASSES
        #         else:  # fasterrcnn/maskrcnn etc
        #             num_categories = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        #         self.category_names = [str(category_id) for category_id in range(num_categories)]
        #         self.category_mapping = {
        #             str(ind): category_name for ind, category_name in enumerate(self.category_names)
        #         }
        # else:
        #     self.category_names = list(self.category_mapping.values())



def loadSwinDetectron(config_path,weights_path,RUN_ON_CPU):
    cfg = LazyConfig.load(config_path)
    if RUN_ON_CPU:
        cfg.train.device = 'cpu'
    cfg.train.init_checkpoint = os.path.join(weights_path)
    cfg.model.roi_heads.box_predictors[-1].test_score_thresh =  0.1

    predictor = Swin_Detectron_Predictor(cfg)
    return predictor

def loadSahiSwinDetectron(config_path,weights_path,RUN_ON_CPU):
    device = 'cuda:0' if not RUN_ON_CPU else 'cpu'
    sahi_detectron_damage_predictor = SahiSwinDetectron(model_path=weights_path,config_path=config_path,
                                                        confidence_threshold=0.1,device=device)
    return sahi_detectron_damage_predictor