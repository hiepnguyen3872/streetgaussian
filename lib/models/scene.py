import os
import torch
from typing import Union
from lib.datasets.dataset import Dataset
from lib.models.gaussian_model import GaussianModel
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.config import cfg
from lib.utils.system_utils import searchForMaxIteration

class Scene:

    gaussians : Union[GaussianModel, StreetGaussianModel]
    dataset: Dataset

    def __init__(self, gaussians: Union[GaussianModel, StreetGaussianModel], dataset: Dataset, init_from_craft: bool = True):
        self.dataset = dataset
        self.gaussians = gaussians
        
        if cfg.mode == 'train':
            point_cloud = self.dataset.scene_info.point_cloud
            scene_raidus = self.dataset.scene_info.metadata['scene_radius']
            print("Creating gaussian model from point cloud")
            self.gaussians.create_from_pcd(point_cloud, scene_raidus)
            
            train_cameras = self.getTrainCameras()
            self.train_cameras_id_to_index = dict()
            for i, train_camera in enumerate(train_cameras):
                self.train_cameras_id_to_index[train_camera.id] = i
            
            pseudo_train_cameras = self.getPseudoTrainCameras()
            self.pseudo_train_cameras_id_to_index = dict()
            for i, pseudo_train_camera in enumerate(pseudo_train_cameras):
                self.pseudo_train_cameras_id_to_index[pseudo_train_camera.id] = i
            
        else:
            # First check if there is a point cloud saved and get the iteration to load from
            # assert(os.path.exists(cfg.point_cloud_dir))
            if cfg.loaded_iter == -1:
                self.loaded_iter = searchForMaxIteration(cfg.point_cloud_dir)
            else:
                self.loaded_iter = cfg.loaded_iter
            # self.loaded_iter = 60000
            # self.loaded_iter = 7000

            # Load pointcloud
            # print("Loading saved pointcloud at iteration {}".format(self.loaded_iter))
            # point_cloud_path = os.path.join(cfg.point_cloud_dir, f"iteration_{str(self.loaded_iter)}/point_cloud.ply")
            
            # self.gaussians.load_ply(point_cloud_path)
            
            # Load checkpoint if it exists (this loads other parameters like the optimized tracking poses)
            self.gaussians.load_state_dict(trained_model_dir=cfg.trained_model_dir, loaded_iter=self.loaded_iter )
            
    def save(self, iteration):
        os.makedirs(cfg.point_cloud_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.point_cloud_dir, f"iteration_{iteration}"), exist_ok=True)
        point_cloud_path = os.path.join(cfg.point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")
        # torch.save((self.gaussians.background.capture(), iteration), point_cloud_path)
        self.gaussians.save_ply(point_cloud_path)

    def getTrainCameras(self, scale=1):
        return self.dataset.train_cameras[scale]

    def getTestCameras(self, scale=1):
        return self.dataset.test_cameras[scale]
    
    def getPseudoTrainCameras(self, scale=1):
        return self.dataset.pseudo_train_cameras[scale]
    
    def getNovelViewCameras(self, scale=1):
        try:
            return self.dataset.novel_view_cameras[scale]
        except:
            return []