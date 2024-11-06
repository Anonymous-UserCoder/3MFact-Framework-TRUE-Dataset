import os
import cv2
import numpy as np
from typing import List
import glob

import logging




class Time:
    def __init__(self, milliseconds: float):
        self.second, self.millisecond = divmod(milliseconds, 1000)
        self.minute, self.second = divmod(self.second, 60)
        self.hour, self.minute = divmod(self.minute, 60)

    def __str__(self):
        return f'{str(int(self.hour)) + "h-" if self.hour else ""}' \
               f'{str(int(self.minute)) + "m-" if self.minute else ""}' \
               f'{str(int(self.second)) + "s-" if self.second else ""}' \
               f'{str(int(self.millisecond)) + "ms"}'

class Frame:
    def __init__(self, no: int, hist: List):
        self.no = no
        self.hist = hist

class FrameCluster:
    def __init__(self, cluster: List[Frame], center: Frame):
        self.cluster = cluster
        self.center = center

    def re_center(self):
        hist_sum = [0] * len(self.cluster[0].hist)
        for i in range(len(self.cluster[0].hist)):
            for j in range(len(self.cluster)):
                hist_sum[i] += self.cluster[j].hist[i]
        self.center.hist = [i / len(self.cluster) for i in hist_sum]

    def keyframe_no(self) -> int:
        no = self.cluster[0].no
        max_similar = 0
        for frame in self.cluster:
            similar = similarity(frame.hist, self.center.hist)
            if similar > max_similar:
                max_similar, no = similar, frame.no
        return no

def similarity(frame1, frame2):
    s = np.vstack((frame1, frame2)).min(axis=0)
    similar = np.sum(s)
    return similar

# def extract_keyframes(video_path: str) -> None:
#     # 初始阈值
#     threshold = float(0.85)
#     # 提取视频文件名和路径
#     video_dir, video_name = os.path.split(video_path)
#     video_base_name = os.path.splitext(video_name)[0]
#     target_path = os.path.join(video_dir, video_base_name)
    
#     # 创建目标文件夹
#     if not os.path.exists(target_path):
#         os.mkdir(target_path)
    
#     adjust = True
#     while adjust:
#         frames = handle_video_frames(video_path)
#         clusters = frames_cluster(frames, threshold)
#         # 清空目标文件夹
#         for file in os.listdir(target_path):
#             os.remove(os.path.join(target_path, file))
#         store_keyframe(video_path, target_path, clusters)

#         num_images = len(os.listdir(target_path))
#         if 4 <= num_images <= 7:
#             adjust = False
#         elif num_images < 4:
#             threshold += 0.005
#         else:
#             threshold -= 0.005
        
#         print(f'阈值：{threshold}，关键帧数量：{num_images}')


def extract_keyframes_with_binary_search(video_path: str, min_keyframes=4, max_keyframes=6, min_threshold=0.3, max_threshold=0.99, max_iterations=20) -> None:
    video_dir, video_name = os.path.split(video_path)
    video_base_name = os.path.splitext(video_name)[0]
    target_path = os.path.join(video_dir, video_base_name)
    
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    left, right = min_threshold, max_threshold
    optimal_threshold = (left + right) / 2

    for _ in range(max_iterations):
        threshold = (left + right) / 2
        frames = handle_video_frames(video_path)
        clusters = frames_cluster(frames, threshold)
        
        # 清空目标文件夹
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
        store_keyframe(video_path, target_path, clusters)

        num_images = len(os.listdir(target_path))
        print(f'阈值：{threshold}，关键帧数量：{num_images}')

        if min_keyframes <= num_images <= max_keyframes:
            optimal_threshold = threshold
            break  # 找到合适的阈值，退出循环
        elif num_images < min_keyframes:
            left = threshold  # 关键帧数量过少，需要降低阈值
        else:
            right = threshold  # 关键帧数量过多，需要提高阈值
        

    print(f"最优阈值：{optimal_threshold}，关键帧数量：{num_images}")

    return target_path
# 注意：这个解决方案中，我们假设关键帧数量与阈值之间存在单调关系。
# 实际应用时，这个关系可能会更复杂，因此可能需要根据实际情况进行调整。


# 处理视频帧函数、聚类函数和存储关键帧函数代码与之前相同，但需要适当修改以适应新的结构和逻辑

def handle_video_frames(video_path: str) -> List[Frame]:
    """
    处理视频获取所有帧的HSV直方图
    :param video_path: 视频路径
    :return: 帧对象数组
    """
    # 创建视频对象
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 帧高，宽
    # print(f'Frames Per Second: {fps}')
    # print(f'Number of Frames : {frame_count}')
    # print(f'Height of Video: {height}')
    # print(f'Width of Video: {width}')

    no = 1
    frames = list()

    # 读取视频帧
    nex, frame = cap.read()
    while nex:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR -> HSV 转换颜色空间
        # 统计颜色直方图，[h,s,v]:[0,1,2] 分为 [12,5,5]份
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 5, 5], [0, 256, 0, 256, 0, 256])
        # numpy 3维数组扁平化
        flatten_hists = hist.flatten()
        # 求均值
        flatten_hists /= height * width
        frames.append(Frame(no, flatten_hists))

        # 显示3个通道的颜色直方图
        # plt.plot(flatten_hists[:12], label=f'H_{times}', color='blue')
        # plt.plot(flatten_hists[12:17], label=f'S_{times}', color='green')
        # plt.plot(flatten_hists[17:], label=f'V_{times}', color='red')
        # plt.legend(loc='best')
        # plt.xlim([0, 22])

        no += 1
        nex, frame = cap.read()

        # plt.show()

    # 释放
    cap.release()
    return frames

def frames_cluster(frames: List[Frame], threshold: float) -> List[FrameCluster]:
    ret_clusters = [FrameCluster([frames[0]], frames[0])]  # 初始化第一个帧为一个聚类的中心
    for frame in frames[1:]:
        max_ratio, clu_idx = 0, -1
        for i, clu in enumerate(ret_clusters):
            sim_ratio = similarity(frame.hist, clu.center.hist)
            if sim_ratio > max_ratio:
                max_ratio, clu_idx = sim_ratio, i

        if max_ratio < threshold:
            ret_clusters.append(FrameCluster([frame], frame))
        else:
            ret_clusters[clu_idx].cluster.append(frame)
            ret_clusters[clu_idx].re_center()

    return ret_clusters


def store_keyframe(video_path: str, target_path: str, frame_clusters: List[FrameCluster]) -> None:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    keyframe_nos = set([cluster.keyframe_no() for cluster in frame_clusters])  # 获取所有关键帧的编号

    no, saved_count = 1, 1  # `no`是当前帧编号，`saved_count`是已保存的关键帧计数器
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if no in keyframe_nos:
            cv2.imwrite(f'{target_path}/{saved_count}.jpg', frame)  # 保存关键帧
            saved_count += 1
        no += 1

    cap.release()




# if __name__ == '__main__':
#     video_path = r"E:\aim\LLM_code\pipeline\extract\KeyFramesExtraction\standard.mp4"
    # target_path = extract_keyframes_with_binary_search(video_path)
    
#     print(f"关键帧已保存至：{target_path}")
#     # 调用示例
#     # optimal_threshold = find_optimal_threshold(video_path, 4, 7)
#     # print(f"Optimal Threshold: {optimal_threshold}")



from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
 
# For windows, the below if condition is must.
def katna_keyframes_extraction(video_file_path, no_of_frames_to_returned):
    # initialize video module
    vd = Video()
    
    video_dir, video_name = os.path.split(video_file_path)
    video_base_name = os.path.splitext(video_name)[0]
    target_path = os.path.join(video_dir, video_base_name)
    # number of images to be returned
    # no_of_frames_to_returned = 12
 
    # initialize diskwriter to save data at desired location
    # disk_writer = KeyFrameDiskWriter(location=r'/home/public/FakeNews/code/example/')



    # Check if target_path exists and has the desired number of frames
    if os.path.exists(target_path) and len([f for f in os.listdir(target_path) if f.endswith('.jpeg')]) >= no_of_frames_to_returned:
        print(f"Keyframes already extracted and present in {target_path}")
        logging.info(f"Keyframes already extracted and present in {target_path}")
        return target_path
    
    disk_writer = KeyFrameDiskWriter(location=target_path)

    # Video file path
    # video_file_path = r"/home/public/FakeNews/code/example/v__jV5sAOOHLk.mp4"
 
    print(f"Input video file path = {video_file_path}")
    logging.info(f"Input video file path = {video_file_path}")
 
    # extract keyframes and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
        writer=disk_writer
    )
    print(f"video {video_base_name}：Keyframes extracted successfully")
    logging.info(f"video {video_base_name}：Keyframes extracted successfully")

 # Reorder and rename extracted keyframes
    reorder_and_rename_images(target_path)
    
    return target_path

def reorder_and_rename_images(directory_path):
    # Find all image files in the directory, assuming they're in the JPG format
    # You can adjust the pattern to match your file types, e.g., "*.png" for PNG files
    images = sorted(glob.glob(os.path.join(directory_path, "*.jpeg")), key=os.path.getmtime)
    
    # Rename images in order
    for i, image_path in enumerate(images, start=1):
        new_name = os.path.join(directory_path, f"{i}.jpeg")
        os.rename(image_path, new_name)
    
    print("Images have been renamed successfully.")
    logging.info("Images have been renamed successfully.")

# video_path = r"/home/public/FakeNews/code/example/v__jV5sAOOHLk.mp4"
# target_path = katna_keyframes_extraction(video_path, 6)