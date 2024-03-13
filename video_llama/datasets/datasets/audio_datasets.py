import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate

class AudioDataset(BaseDataset):
    def __init__(self):
        """
        aud_root (string): Root directory of audio (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        # 读取一个路径下所有的
        ts_df = []
        for file_name in os.listdir(ann_root):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(ann_root, file_name))
                ts_df.append(df)

        merged_df = pd.concat(ts_df)

        self.annotation = merged_df
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 8
        self.frm_sampling_strategy = 'headtail'


    def _get_audio_path(self, sample):
        rel_audio_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_audio_fp = os.path.join(self.vis_root,  rel_audio_fp)
        return full_audio_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()
            video_id = sample_dict['videoid']

            if 'name' in sample_dict.keys():
                text = sample_dict['name'].strip()
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch video
            video_path = self._get_video_path(sample_dict)
            # if os.path.exists(video_path):
            try:
                video = self.vis_processor(video_path)
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            caption = self.text_processor(text)

            # print(video.size())
            if video is None or caption is None \
                    or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": caption,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)