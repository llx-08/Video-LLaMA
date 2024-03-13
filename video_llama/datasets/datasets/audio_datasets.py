import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from video_llama.models.ImageBind.data import load_and_transform_audio_data

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
            try:
                sample = self.annotation[index]

                logging.info("==================================check sample==================================")
                logging.info(sample)

                audio_path = self._get_audio_path(sample)
                conversation_list = sample['QA']

                audio, msg = load_and_transform_audio_data()

                audio, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling="uniform", return_msg=True
                )
                video = self.transform(video)

                if 'cn' in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None,
                                                cur_token_len=self.num_video_query_token, msg=msg)
                new_sources = convert_source_vicuna_format(sources)

                if self.model_type == 'vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type == 'llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                else:
                    print('not support')
                    raise ('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
            except:
                print(f"Failed to load examples with video: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type": 'video',
        }
    def __len__(self):
        return len(self.annotation)