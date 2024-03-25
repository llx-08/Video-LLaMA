import torch
from torch import nn
from torch.nn import functional as F

# 假设我们有一个预训练的模型
class YourModel(Blip2Base):
    def __init__(self):
        super().__init__()
        # 定义你的模型结构
        self.audio_Qformer, self.audio_query_tokens = self.init_video_Qformer(
            num_query_token=self.num_audio_query_token, \
            vision_width=self.audio_hidden_size, num_hidden_layers=2)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

    def forward(self, x):
        pass

# 实例化模型
model = YourModel()

# 加载预训练的模型checkpoint
checkpoint = torch.load('/lilinxuan/llx_videollama/video_llama/output'
                        '/audiobranch_stage2_finetune/20240324150/checkpoint_199.pth')
print("dict")
print(checkpoint)

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