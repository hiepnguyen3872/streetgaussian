import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.optical_flow_utils import calculate_gs_optical_flow, flow_to_image
# cfg.model_path = os.path.join(cfg.workspace, 'street_gaussian_wo_3DOD_lidar_sparse_10_frames/fine', cfg.task, cfg.exp_name)
# cfg.trained_model_dir = os.path.join(cfg.model_path, 'trained_model')
# cfg.point_cloud_dir = os.path.join(cfg.model_path, 'point_cloud')
import cv2

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False
    # cfg.model.nsg.use_deformation_model = False
    psnr_train = 0.0
    psnr_test = 0.0
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        # if True: 
        #     save_dir = os.path.join(cfg.model_path, 'pseudo_train', "ours_{}".format(scene.loaded_iter))
        #     visualizer = Visualizer(save_dir)
        #     cameras = scene.getPseudoTrainCameras()
        #     for idx, camera in enumerate(tqdm(cameras, desc="Rendering Pseudo Training View")):
                
        #         torch.cuda.synchronize()
        #         start_time = time.time()
        #         result = renderer.render(camera, gaussians, use_hexplane = cfg.model.nsg.use_deformation_model)
                
        #         torch.cuda.synchronize()
        #         end_time = time.time()
        #         mask = torch.ones_like(camera.original_image[0]).bool().cuda()
        #         times.append((end_time - start_time) * 1000)
        #         # image = torch.clamp(result['rgb'], 0.0, 1.0)
        #         # gt_image = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)
        #         # psnr_train += psnr(image, gt_image, mask).mean().double()
                
        #         visualizer.visualize(result, camera, cfg.data.use_semantic, pseudo_view = True)
        # all_cameras = sorted(scene.getTrainCameras() + scene.getTestCameras(), key=lambda camera: camera.raw_time)

        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians, use_hexplane = cfg.model.nsg.use_deformation_model)
                
                torch.cuda.synchronize()
                end_time = time.time()
                mask = torch.ones_like(camera.original_image[0]).bool().cuda()
                times.append((end_time - start_time) * 1000)
                # image = torch.clamp(result['rgb'], 0.0, 1.0)
                # gt_image = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)
                # psnr_train += psnr(image, gt_image, mask).mean().double()
                
                visualizer.visualize(result, camera, cfg.data.use_semantic)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians, use_hexplane = cfg.model.nsg.use_deformation_model)
                torch.cuda.synchronize()
                end_time = time.time()
                
                mask = torch.ones_like(camera.original_image[0]).bool().cuda()
                times.append((end_time - start_time) * 1000)
                
                image = torch.clamp(result['rgb'], 0.0, 1.0)
                gt_image = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)
                psnr_test += psnr(image, gt_image, mask).mean().double()
                
                visualizer.visualize(result, camera, cfg.data.use_semantic)
            
            # for idx, camera in enumerate(tqdm(all_cameras, desc="Rendering Testing View")):
            #     if idx == len(all_cameras) - 1: 
            #         break
            #     torch.cuda.synchronize()
            #     start_time = time.time()
                
            #     result = renderer.render(camera, gaussians, use_hexplane = cfg.model.nsg.use_deformation_model)
            #     new_camera = all_cameras[idx + 1]
            #     result_t2 = renderer.render(new_camera, gaussians, use_hexplane = cfg.model.nsg.use_deformation_model)
            #     torch.cuda.synchronize()
            #     gs_optical_flow = calculate_gs_optical_flow(result, result_t2)
            #     gs_optical_flow_viz = flow_to_image(gs_optical_flow)
            #     name = camera.image_name
            #     cv2.imwrite(f"{visualizer.result_dir}/{name}_flow.jpg", gs_optical_flow_viz)
            #     end_time = time.time()
            #     times.append((end_time - start_time) * 1000)
                
            #     mask = torch.ones_like(camera.original_image[0]).bool().cuda()
            #     times.append((end_time - start_time) * 1000)
                
            #     image = torch.clamp(result['rgb'], 0.0, 1.0)
            #     gt_image = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)
            #     psnr_test += psnr(image, gt_image, mask).mean().double()
                
        
        print(times)       
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
        print('psnr test: ', psnr_test / len(cameras))
                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
