import torch
from torch import nn
from torch.nn import functional as F
from video_llama.models.blip2 import Blip2Base, disabled_train

# 加载预训练的模型checkpoint
checkpoint = torch.load('/home/asr/lilinxuan/llx_videollama/video_llama/output'
                        '/audiobranch_stage2_finetune/20240324150/checkpoint_199.pth')
print("dict")
print(checkpoint)

# 假设我们有一个预训练的模型
class YourModel(Blip2Base):
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    def __init__(self,
                 vit_model="eva_clip_g",
                 img_size=224,
                 drop_path_rate=0,
                 use_grad_checkpoint=False,
                 vit_precision="fp16",
                 num_query_token=32,
                 num_audio_query_token = 8,):
        super().__init__()

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.ln_vision.train = disabled_train
        logging.info("freeze vision encoder")


        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        # llama linear
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

    def forward(self, x):
        pass

# 实例化模型
model = YourModel()


# 假设checkpoint包含模型的状态字典和可能的其他信息，如优化器状态
model.load_state_dict(checkpoint['model_state_dict'])

for name, param in model.named_parameters():
    print(name)

# 分割模型参数
audio_Qformer = {name: param for name, param in model.named_parameters() if 'encoder' in name}
llama_proj = {name: param for name, param in model.named_parameters() if 'decoder' in name}

audio_Qformer = nn.Sequential(*[nn.ParameterDict(list(params.items())) for params in [audio_Qformer]])

torch.save(audio_Qformer.state_dict(), 'audio_Qformer.pth')

llama_proj = nn.Sequential(*[nn.ParameterDict(list(params.items())) for params in [llama_proj]])

torch.save(decoder.state_dict(), 'llama_proj.pth')